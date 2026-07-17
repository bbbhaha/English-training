"""Web demo for the project end-to-end pronunciation diagnosis pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
import json
import math
import mimetypes
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


APP_VERSION = "PHONE_THREE_STATE_V5_IPA"
print(f"========== RUNNING NEW APP VERSION: {APP_VERSION} ==========", flush=True)

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from phase15_verification.analysis import add_evidence_columns  # noqa: E402
from pronunciation.alignment import align_audio_to_text, save_alignment_csv  # noqa: E402
from pronunciation.g2p import write_g2p_json  # noqa: E402
from predict_pronunciation import (  # noqa: E402
    _add_verifier_defaults,
    _apply_manual_calibrator,
    _final_output,
    _load_config,
    _prediction_frame,
    _prepare_audio_for_alignment,
    _score_phase1,
    ensure_prediction_coverage,
)
from pronunciation.ctc_phone_diagnosis import (  # noqa: E402
    DEFAULT_PHONE_CTC_MODEL,
    DEFAULT_REFERENCE_PHONE_CTC_MODEL,
    DEFAULT_THREE_STATE_MODEL,
    apply_phone_three_state_model,
    force_confirmed_word_deletions,
    score_audio_phones_ctc,
    summarize_three_state_phones,
)
from pronunciation.ctc_word_deletion import score_audio_word_deletions  # noqa: E402
from pronunciation.final_word_decision import run_word_level_diagnosis  # noqa: E402
from pronunciation.mandarin_deletion_fusion import DEFAULT_MODEL_PATH as DEFAULT_MANDARIN_DELETION_MODEL  # noqa: E402
from pronunciation.target_words import build_target_word_table, ensure_word_summary_coverage  # noqa: E402
from pronunciation.text_audio_consistency import check_text_audio_consistency  # noqa: E402


WEB_OUTPUT_DIR = ROOT / "outputs" / "webapp"
DISPLAY_COLUMNS = [
    "word",
    "target_phone",
    "start_ms",
    "end_ms",
    "phone_error_percent",
    "phone_state",
    "phone_state_zh",
    "phone_state_confidence",
    "phone_probability_correct",
    "phone_probability_mispronounced",
    "phone_probability_deleted",
    "recognized_phone",
    "reference_recognized_phone",
    "reference_ctc_deletion_margin",
    "reference_deletion_supported",
    "phone_decision",
    "phone_error_type",
    "alignment_quality",
    "evidence_summary",
    "display_decision",
    "display_error",
    "display_align",
    "display_error_type",
    "lexicon_status",
    "deletion_decision",
    "deletion_score",
    "asr_missing_word",
    "asr_missing_confidence",
    "ctc_deletion_score",
    "ctc_deletion_available",
]


@dataclass
class UploadedFile:
    filename: str
    content: bytes


def json_response(handler: BaseHTTPRequestHandler, payload: dict[str, object], status: int = 200) -> None:
    data = json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
    handler.send_header("Pragma", "no-cache")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _json_safe(value: object) -> object:
    """Convert pandas/numpy missing values into strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def file_response(handler: BaseHTTPRequestHandler, path: Path) -> None:
    if not path.exists() or not path.is_file():
        handler.send_error(404)
        return
    content = path.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
    handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
    handler.send_header("Pragma", "no-cache")
    handler.send_header("Content-Length", str(len(content)))
    handler.end_headers()
    handler.wfile.write(content)


def diagnose(
    audio_path: Path,
    text: str,
    utterance_id: str,
    speaker_id: str,
    trim_silence: bool = True,
) -> dict[str, object]:
    WEB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    session_id = f"session_{int(time.time() * 1000)}_{_safe_file_stem(utterance_id or 'web')}"
    session_dir = WEB_OUTPUT_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    output = session_dir / "prediction.csv"
    alignment_output = session_dir / "alignment.csv"
    g2p_output = session_dir / "g2p.json"
    word_summary_output = session_dir / "word_summary.csv"
    preprocessed_output = session_dir / "audio_16k.wav"
    args = argparse.Namespace(
        audio=audio_path,
        text=text,
        output=output,
        utterance_id=utterance_id,
        speaker_id=speaker_id,
        speaker_gender="",
        speaker_age=0.0,
        alignment_output=alignment_output,
        g2p_output=g2p_output,
        alignment_models=ROOT / "artifacts" / "baseline_acoustic_v1" / "phone_gaussians.joblib",
        phase1_model=ROOT / "artifacts" / "phase1_acoustic_fusion_macro_models" / "feature_logreg.joblib",
        phase15_config=ROOT / "configs" / "phase15" / "aggregator.yaml",
        manual_calibrator=ROOT / "outputs" / "phase15_verification" / "manual_calibration_v2" / "manual_calibrated_verifier.joblib",
        true_error_threshold=None,
        main_error_threshold=0.05,
        decision_mode="phone_diagnosis",
        detect_deletion_as_error=False,
        word_summary_output=word_summary_output,
        preprocessed_audio_output=preprocessed_output,
        no_auto_preprocess=False,
        trim_silence=trim_silence,
        enable_asr=True,
        enable_asr_consistency_check=True,
        asr_transcript=None,
        asr_model="auto",
        mandarin_deletion_model=DEFAULT_MANDARIN_DELETION_MODEL,
        phone_ctc_model=DEFAULT_PHONE_CTC_MODEL,
        reference_phone_ctc_model=DEFAULT_REFERENCE_PHONE_CTC_MODEL,
        phone_three_state_model=DEFAULT_THREE_STATE_MODEL,
        allow_phone_model_download=False,
    )

    target_word_table = build_target_word_table(text, utterance_id=utterance_id)
    audio_for_alignment = _prepare_audio_for_alignment(args)
    alignment, g2p = align_audio_to_text(
        audio_for_alignment,
        text=text,
        models_path=args.alignment_models,
        target_word_table=target_word_table,
    )
    save_alignment_csv(alignment, alignment_output)
    if g2p is not None:
        write_g2p_json(g2p, g2p_output)

    frame = _prediction_frame(alignment, args)
    frame = _score_phase1(frame, args.phase1_model)
    frame = _add_verifier_defaults(frame)
    frame = add_evidence_columns(frame, _load_config(args.phase15_config))
    frame = _apply_manual_calibrator(frame, args.manual_calibrator, args.true_error_threshold)
    out = _final_output(frame, args)
    out = ensure_prediction_coverage(out, target_word_table, g2p)
    phone_evidence = score_audio_phones_ctc(
        audio_for_alignment,
        out,
        model_id=args.phone_ctc_model,
        local_files_only=not args.allow_phone_model_download,
    )
    reference_phone_evidence = score_audio_phones_ctc(
        audio_for_alignment,
        out,
        model_id=args.reference_phone_ctc_model,
        local_files_only=not args.allow_phone_model_download,
    )
    out = apply_phone_three_state_model(
        out,
        phone_evidence,
        classifier_path=args.phone_three_state_model,
        reference_evidence=reference_phone_evidence,
    )
    word_summary = summarize_three_state_phones(out)
    word_summary = ensure_word_summary_coverage(target_word_table, word_summary)
    consistency, consistency_meta = check_text_audio_consistency(
        audio_path=audio_for_alignment,
        target_text=text,
        asr_model="faster_whisper",
    )
    ctc_deletion_features = score_audio_word_deletions(
        audio_for_alignment,
        text,
        local_files_only=True,
    )
    word_summary = run_word_level_diagnosis(
        out,
        word_summary,
        consistency,
        ctc_deletion_features,
        args.mandarin_deletion_model,
    )
    word_summary["asr_transcript"] = str(consistency_meta.get("asr_transcript", ""))
    word_summary["asr_available"] = bool(consistency_meta.get("asr_available", False))
    word_summary = ensure_word_summary_coverage(target_word_table, word_summary)
    out = force_confirmed_word_deletions(out, word_summary)
    out.to_csv(output, index=False, encoding="utf-8-sig")
    word_summary.to_csv(word_summary_output, index=False, encoding="utf-8-sig")

    prediction_df = pd.read_csv(output, encoding="utf-8-sig")
    word_summary_df = pd.read_csv(word_summary_output, encoding="utf-8-sig")
    display_frame = apply_phone_diagnosis_display(prediction_df, word_summary_df)
    print("prediction path:", output, flush=True)
    print("word_summary path:", word_summary_output, flush=True)
    print("prediction decision counts:", flush=True)
    print(prediction_df["decision"].value_counts(dropna=False), flush=True)
    print("word_summary:", flush=True)
    print(word_summary_df.fillna("").to_string(), flush=True)
    print("display_df:", flush=True)
    print(
        display_frame[
            [
                "word",
                "word_index",
                "target_phone",
                "display_decision",
                "display_error",
                "display_align",
                "display_error_type",
                "phone_error_probability",
            ]
        ].to_string(),
        flush=True,
    )
    rows = display_frame[DISPLAY_COLUMNS].fillna("").to_dict(orient="records")
    counts = display_frame["phone_decision"].value_counts().to_dict()
    debug_payload = _prediction_debug_payload(
        args.decision_mode,
        output,
        word_summary_output,
        prediction_df,
        word_summary_df,
        display_frame,
        rows,
    )
    print(json.dumps(_json_safe(debug_payload), ensure_ascii=False, allow_nan=False, indent=2), flush=True)
    return {
        "utterance_id": utterance_id,
        "speaker_id": speaker_id,
        "text": text,
        "n_phones": len(rows),
        "decision_counts": counts,
        "n_correct": int(counts.get("correct", 0)),
        "n_acceptable_accent": 0,
        "n_true_error": int(counts.get("true_error", 0)),
        "n_uncertain_review": int(counts.get("uncertain_review", 0)),
        "rows": rows,
        "debug": debug_payload,
        "app_version": APP_VERSION,
        "artifacts": {
            "prediction_csv": str(output),
            "alignment_csv": str(alignment_output),
            "g2p_json": str(g2p_output),
            "word_summary_csv": str(word_summary_output),
            "preprocessed_audio": str(preprocessed_output) if preprocessed_output.exists() else "",
        },
    }


class AppHandler(BaseHTTPRequestHandler):
    server_version = "PronunciationE2EDemo/0.2"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            file_response(self, STATIC_DIR / "index.html")
            return
        if parsed.path.startswith("/static/"):
            rel = parsed.path.removeprefix("/static/").replace("/", "\\")
            file_response(self, STATIC_DIR / rel)
            return
        if parsed.path == "/api/prompts":
            json_response(
                self,
                {
                    "prompts": [
                        {"label": "MAGA", "text": "MAKE AMERICA GREAT AGAIN"},
                        {"label": "demo", "text": "MIKE LIKES THE ORANGE ONE"},
                        {"label": "bear", "text": "WE CALL IT BEAR"},
                        {"label": "sentence", "text": "SHE SEES THE BLUE BIRD"},
                    ]
                },
            )
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/diagnose":
            self.send_error(404)
            return
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            json_response(self, {"error": "Expected multipart/form-data."}, status=400)
            return
        form, files = _parse_multipart(self, content_type)
        text = str(form.get("text", "")).strip()
        utterance_id = str(form.get("utterance_id", f"web_{int(time.time())}")).strip() or "web_demo"
        speaker_id = str(form.get("speaker_id", "web_user")).strip() or "web_user"
        trim_silence = str(form.get("trim_silence", "1")).strip() != "0"
        audio_item = files.get("audio")

        if not text:
            json_response(self, {"error": "Text is required."}, status=400)
            return
        if audio_item is None or not audio_item.content:
            json_response(self, {"error": "Audio file is required."}, status=400)
            return
        suffix = _uploaded_suffix(audio_item.filename)
        with tempfile.NamedTemporaryFile(prefix="pron_e2e_", suffix=suffix, delete=False) as tmp:
            tmp.write(audio_item.content)
            tmp_path = Path(tmp.name)
        try:
            payload = diagnose(
                tmp_path,
                text,
                utterance_id,
                speaker_id,
                trim_silence=trim_silence,
            )
        except Exception as exc:
            json_response(self, {"error": str(exc)}, status=500)
            return
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        json_response(self, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the end-to-end pronunciation diagnosis web demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"Pronunciation E2E demo running at http://{args.host}:{args.port}")
    server.serve_forever()


def _row_for_api(row: dict[str, object]) -> dict[str, object]:
    payload = {
        "utterance_id": row.get("utterance_id", ""),
        "speaker_id": row.get("speaker_id", ""),
        "word": row.get("word", ""),
        "word_index": _number(row.get("word_index", 0), int),
        "target_phone": row.get("target_phone", ""),
        "phone_index": _number(row.get("phone_index", 0), int),
        "start_ms": _number(row.get("start_ms", 0.0), float),
        "end_ms": _number(row.get("end_ms", 0.0), float),
        "duration_ms": _number(row.get("duration_ms", 0.0), float),
        "model_error_score": _number(row.get("model_error_score", 0.0), float),
        "prob_correct": _number(row.get("prob_correct", 0.0), float),
        "manual_calibrated_error_probability": _number(row.get("manual_calibrated_error_probability", 0.0), float),
        "decision": row.get("decision", "uncertain_review"),
        "confidence": _number(row.get("confidence", 0.0), float),
        "error_type": row.get("error_type", ""),
        "word_decision": row.get("word_decision", ""),
        "word_error_type": row.get("word_error_type", ""),
        "word_alignment_quality": row.get("word_alignment_quality", ""),
        "alignment_quality": row.get("alignment_quality", ""),
        "review_reason": row.get("review_reason", ""),
        "g2p_source": row.get("g2p_source", ""),
        "lexicon_status": row.get("lexicon_status", ""),
        "g2p_confidence": row.get("g2p_confidence", ""),
        "lexicon_display": "自动推测发音"
        if str(row.get("lexicon_status", "")) in {"g2p_en", "phonemizer"}
        else str(row.get("lexicon_status", "")),
        "possible_missing_word": _boolish(row.get("possible_missing_word", False)),
        "missing_word_reason": row.get("missing_word_reason", ""),
        "deletion_trigger_source": row.get("deletion_trigger_source", "none"),
        "debug_reason": row.get("debug_reason", ""),
        "display_decision": row.get("display_decision", ""),
        "display_error": row.get("display_error", ""),
        "display_align": row.get("display_align", ""),
        "display_error_type": row.get("display_error_type", ""),
    }
    payload.update(deletion_only_display_fields(payload))
    return payload


def apply_phone_diagnosis_display(
    prediction_df: pd.DataFrame,
    word_summary_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one frontend row per prediction phone without word aggregation."""
    display = prediction_df.copy()
    defaults: dict[str, object] = {
        "word": "",
        "target_phone": "<UNK>",
        "start_ms": float("nan"),
        "end_ms": float("nan"),
        "alignment_quality": "bad",
        "phone_error_probability": 0.5,
        "phone_error_percent": 50.0,
        "phone_decision": "uncertain_review",
        "phone_error_type": "",
        "phone_confidence": 0.0,
        "evidence_summary": "Phone-level diagnosis is unavailable.",
    }
    for column, default in defaults.items():
        if column not in display.columns:
            display[column] = default
        else:
            display[column] = display[column].fillna(default)

    if word_summary_df is not None and not word_summary_df.empty and "word_index" in display.columns:
        summary = word_summary_df.copy()
        display["word_index"] = display["word_index"].astype(str)
        summary["word_index"] = summary["word_index"].astype(str)
        summary = summary.rename(
            columns={
                "alignment_quality": "word_alignment_quality",
                "error_type": "word_error_type",
                "evidence_summary": "word_evidence_summary",
            }
        )
        word_fields = [
            "word_index",
            "deletion_decision",
            "deletion_score",
            "final_word_decision",
            "final_error_type",
            "word_error_type",
            "word_alignment_quality",
            "asr_missing_word",
            "asr_missing_confidence",
            "ctc_deletion_score",
            "ctc_deletion_available",
            "word_evidence_summary",
        ]
        available = [column for column in word_fields if column in summary.columns]
        display = display.drop(
            columns=[column for column in available if column != "word_index" and column in display.columns]
        )
        display = display.merge(
            summary[available].drop_duplicates("word_index", keep="last"),
            on="word_index",
            how="left",
        )

    if "phone_state" in display.columns:
        display["display_decision"] = display["phone_state"].astype(str).map(
            {"correct": "读对", "mispronounced": "读错", "deleted": "漏读", "unavailable": "单词暂未收录"}
        ).fillna("单词暂未收录")
    else:
        display["display_decision"] = display["phone_decision"].astype(str).map(
            {"correct": "正确", "true_error": "发音错误", "uncertain_review": "需复核"}
        ).fillna("单词暂未收录")
    if "phone_state" in display.columns:
        display["display_error"] = display.apply(format_phone_state_probability, axis=1)
    else:
        display["display_error"] = [
            format_phone_error_display(percent, error_type)
            for percent, error_type in zip(display["phone_error_percent"], display["phone_error_type"])
        ]
    display["display_align"] = display["alignment_quality"].astype(str)
    display["display_error_type"] = display["phone_error_type"].astype(str)
    for index, row in display.iterrows():
        unavailable = (
            _text(row.get("g2p_status")).lower() == "failed"
            or _text(row.get("lexicon_status")).lower() == "failed"
            or _text(row.get("target_phone")).upper() == "<UNK>"
            or _text(row.get("final_error_type")) == "g2p_issue"
        )
        deletion_type = _text(row.get("final_error_type")) or _text(row.get("word_error_type"))
        deletion_decision = _text(row.get("deletion_decision"))
        if unavailable:
            display.loc[index, ["display_decision", "display_error", "display_error_type"]] = [
                "单词暂未收录",
                "无法判断",
                "g2p_issue",
            ]
        elif deletion_type == "deletion" or deletion_decision == "deletion":
            display.loc[index, ["display_decision", "display_error", "display_align", "display_error_type"]] = [
                "漏读",
                "漏读",
                "suspect",
                "deletion",
            ]
        if deletion_type == "deletion" and _text(row.get("word_evidence_summary")):
            display.loc[index, "evidence_summary"] = _text(row.get("word_evidence_summary"))
    return display


def format_phone_state_probability(row: pd.Series) -> str:
    state = _text(row.get("phone_state"))
    labels = {"correct": "读对", "mispronounced": "读错", "deleted": "漏读"}
    probability_columns = {
        "correct": "phone_probability_correct",
        "mispronounced": "phone_probability_mispronounced",
        "deleted": "phone_probability_deleted",
    }
    if state not in labels:
        return "无法判断"
    probability = _float_value(row.get(probability_columns[state]), 0.0)
    return f"{labels[state]} {probability * 100:.0f}%"


def format_phone_error_display(percent: object, error_type: object) -> str:
    value = _float_value(percent, 50.0)
    error = _text(error_type)
    label = {
        "mispronunciation": "读错",
        "deletion": "漏读",
        "possible_mispronunciation": "疑似错误",
        "alignment_issue": "对齐失败",
    }.get(error, "")
    return f"{value:.0f}%" + (f" {label}" if label else "")


def apply_word_summary_display(
    prediction_df: pd.DataFrame,
    word_summary_df: pd.DataFrame,
    decision_mode: str = "deletion_only",
) -> pd.DataFrame:
    """Merge word-level deletion results into independent frontend display fields."""
    print("prediction columns:", prediction_df.columns.tolist(), flush=True)
    print("word_summary columns:", word_summary_df.columns.tolist(), flush=True)
    print("prediction shape:", prediction_df.shape, flush=True)
    print("word_summary shape:", word_summary_df.shape, flush=True)

    prediction = prediction_df.copy()
    summary = word_summary_df.copy()
    if "word_index" not in prediction.columns or "word_index" not in summary.columns:
        raise KeyError("prediction_df and word_summary_df must both contain word_index")

    prediction["word_index"] = prediction["word_index"].astype(str)
    summary["word_index"] = summary["word_index"].astype(str)
    summary = summary.rename(
        columns={
            "error_type": "word_error_type",
            "alignment_quality": "word_alignment_quality",
        }
    )
    summary_columns = [
        "word_index",
        "word",
        "possible_missing_word",
        "word_decision",
        "word_error_type",
        "deletion_trigger_source",
        "missing_word_reason",
        "word_alignment_quality",
        "lexicon_status",
        "g2p_source",
        "g2p_confidence",
        "asr_available",
        "asr_transcript",
        "asr_word",
        "asr_edit_op",
        "asr_word_status",
        "asr_missing_word",
        "asr_substituted_word",
        "text_audio_consistency_status",
        "text_audio_mismatch",
        "text_audio_mismatch_type",
        "text_audio_mismatch_score",
        "deletion_score",
        "mispronunciation_score",
        "alignment_issue_score",
        "final_word_decision",
        "final_decision",
        "final_error_type",
        "final_error_probability",
        "final_error_percent",
        "evidence_summary",
    ]
    for column in summary_columns[1:]:
        if column not in summary.columns:
            summary[column] = ""

    overlapping = [column for column in summary_columns[1:] if column in prediction.columns]
    prediction = prediction.drop(columns=overlapping)
    if prediction.empty:
        phone_details = pd.DataFrame(columns=["word_index"])
    else:
        def join_phones(values: pd.Series) -> str:
            return " ".join(value for value in values.fillna("").astype(str) if value)

        aggregations = {
            column: "first"
            for column in prediction.columns
            if column not in {"word_index", "target_phone", "word"}
        }
        if "target_phone" in prediction.columns:
            aggregations["target_phone"] = join_phones
        phone_details = prediction.groupby("word_index", sort=False, dropna=False).agg(aggregations).reset_index()
    display_df = summary.merge(phone_details, on="word_index", how="left")
    if "word" not in display_df.columns:
        display_df["word"] = ""

    print("display columns:", display_df.columns.tolist(), flush=True)
    debug_columns = [
        "word",
        "word_index",
        "decision",
        "error_type",
        "possible_missing_word",
        "word_decision",
        "word_error_type",
        "deletion_trigger_source",
    ]
    for column in debug_columns:
        if column not in display_df.columns:
            display_df[column] = ""
    print(display_df[debug_columns].to_string(), flush=True)

    display_df["display_decision"] = "正确"
    display_df["display_error"] = "0%"
    display_df["display_align"] = display_df.get(
        "alignment_quality",
        pd.Series("", index=display_df.index),
    ).fillna("").astype(str)
    display_df["display_error_type"] = ""
    display_df["lexicon_display"] = display_df.get(
        "lexicon_status", pd.Series("", index=display_df.index)
    ).fillna("").astype(str).map(
        lambda value: "自动推测发音" if value in {"g2p_en", "phonemizer"} else value
    )

    if decision_mode == "deletion_only":
        for index, row in display_df.iterrows():
            final_type = _text(row.get("final_error_type")) or _text(row.get("word_error_type"))
            final_decision = _text(row.get("final_word_decision")) or _text(row.get("final_decision"))
            possible_missing = _boolish(row.get("possible_missing_word"))
            if not final_type and possible_missing:
                final_type, final_decision = "possible_deletion", "possible_deletion"
            raw_probability = row.get("final_error_probability")
            explicit_probability = _text(raw_probability) != ""
            probability = _float_value(raw_probability, float("nan"))
            if pd.isna(probability):
                probability = {
                    "deletion": 0.95,
                    "possible_deletion": 0.75,
                    "text_audio_mismatch": 0.85,
                    "alignment_issue": 0.50,
                    "g2p_issue": 0.50,
                }.get(final_type, 0.0)
            display_df.loc[index, "display_decision"] = format_final_decision_display(final_decision, final_type)
            if explicit_probability:
                display_error = format_final_error_display(probability, final_type)
            else:
                display_error = {
                    "deletion": "漏读",
                    "possible_deletion": "疑似漏读",
                    "text_audio_mismatch": "文本音频不一致",
                    "alignment_issue": "对齐失败",
                    "g2p_issue": "无法判断",
                }.get(final_type, "0%")
            display_df.loc[index, "display_error"] = display_error
            display_df.loc[index, "display_error_type"] = final_type
            if final_type in {"deletion", "possible_deletion", "text_audio_mismatch"}:
                display_df.loc[index, "display_align"] = "suspect"
            elif final_type in {"alignment_issue", "g2p_issue"}:
                display_df.loc[index, "display_align"] = "bad"

    return display_df


def format_final_error_display(probability: object, error_type: object) -> str:
    error = _text(error_type)
    if error == "g2p_issue":
        return "无法判断"
    value = min(max(_float_value(probability, 0.0), 0.0), 1.0)
    percent = f"{int(round(value * 100))}%"
    labels = {
        "deletion": "漏读",
        "possible_deletion": "疑似漏读",
        "text_audio_mismatch": "文本音频不一致",
        "alignment_issue": "需复核",
        "g2p_issue": "无法判断",
    }
    return "0%" if not error else f"{percent} {labels.get(error, '')}".strip()


def format_final_decision_display(decision: object, error_type: object = "") -> str:
    final = _text(decision)
    error = _text(error_type)
    if error == "deletion" or final == "deletion":
        return "漏读"
    if error == "possible_deletion" or final == "possible_deletion":
        return "疑似漏读/需复核"
    if error == "text_audio_mismatch" or final == "substituted_word":
        return "文本音频不一致"
    if error == "g2p_issue" or final == "g2p_issue":
        return "单词暂未收录"
    if error == "alignment_issue" or final == "uncertain_review":
        return "需复核"
    if final == "mispronounced":
        return "发音错误"
    if final == "acceptable_accent":
        return "可接受口音"
    return "正确"


def _text(value: object) -> str:
    return "" if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)) else str(value).strip()


def _float_value(value: object, default: float) -> float:
    try:
        return default if pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return default


def deletion_only_display_fields(row: dict[str, object]) -> dict[str, str]:
    if row.get("display_error") or row.get("display_decision") or row.get("display_align"):
        display_error_type = str(row.get("display_error_type", ""))
        display_align = str(row.get("display_align", "")) or str(row.get("alignment_quality", ""))
        if display_error_type == "g2p_issue":
            return {"error_display": "无法判断", "decision_display": "单词暂未收录", "align_display": "bad"}
        if display_error_type == "deletion":
            return {"error_display": "漏读", "decision_display": "漏读", "align_display": "suspect"}
        if display_error_type == "possible_deletion":
            return {"error_display": "疑似漏读", "decision_display": "疑似漏读/需复核", "align_display": "suspect"}
        if display_error_type == "alignment_issue" or display_align.lower() in {"bad", "failed", "alignment_failed"}:
            return {"error_display": "对齐失败", "decision_display": "需复核", "align_display": "bad"}
        return {
            "error_display": str(row.get("display_error", "")) or "0%",
            "decision_display": str(row.get("display_decision", "")) or "正确",
            "align_display": display_align,
        }
    decision = str(row.get("decision", ""))
    error_type = str(row.get("error_type", ""))
    alignment_quality = str(row.get("alignment_quality", ""))
    if error_type == "g2p_issue":
        error_display = "无法判断"
        decision_display = "单词暂未收录"
        alignment_quality = "bad"
    elif error_type == "alignment_issue" or alignment_quality.lower() in {"bad", "failed", "alignment_failed"}:
        error_display = "对齐失败"
        decision_display = "需复核"
        alignment_quality = "bad"
    elif decision == "correct":
        error_display = "0%"
        decision_display = "正确"
    elif decision == "uncertain_review" and error_type == "possible_deletion":
        error_display = "疑似漏读"
        decision_display = "疑似漏读/需复核"
    elif decision == "true_error" and error_type == "deletion":
        error_display = "漏读"
        decision_display = "漏读"
    else:
        error_display = "0%"
        decision_display = "需复核" if decision == "uncertain_review" else ("正确" if decision == "correct" else decision)
    return {
        "error_display": error_display,
        "decision_display": decision_display,
        "align_display": alignment_quality,
    }


def _prediction_debug_payload(
    decision_mode: str,
    prediction_path: Path,
    word_summary_path: Path,
    prediction: pd.DataFrame,
    word_summary: pd.DataFrame,
    display_frame: pd.DataFrame,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    error_prob = pd.to_numeric(
        prediction.get("manual_calibrated_error_probability", pd.Series(dtype=float)),
        errors="coerce",
    )
    warning = ""
    if (
        "possible_missing_word" in word_summary.columns
        and word_summary["possible_missing_word"].astype(str).str.lower().isin({"true", "1", "yes"}).any()
        and rows
        and all(str(row.get("display_decision", "")) == "正确" for row in rows)
    ):
        warning = "WARNING: word_summary indicates deletion but frontend display shows all correct."
    return {
        "decision_mode": decision_mode,
        "prediction_csv": str(prediction_path),
        "word_summary_csv": str(word_summary_path),
        "decision_counts": prediction.get("decision", pd.Series(dtype=str)).astype(str).value_counts().to_dict(),
        "alignment_quality_counts": prediction.get("alignment_quality", pd.Series(dtype=str)).astype(str).value_counts().to_dict(),
        "manual_calibrated_error_probability_describe": error_prob.describe().fillna(0).to_dict() if len(error_prob) else {},
        "word_summary_rows": word_summary.to_dict(orient="records"),
        "word_decision_counts": word_summary.get("word_decision", pd.Series(dtype=str)).astype(str).value_counts().to_dict(),
        "word_error_type_counts": word_summary.get("error_type", pd.Series(dtype=str)).astype(str).value_counts().to_dict(),
        "word_possible_missing_counts": word_summary.get("possible_missing_word", pd.Series(dtype=str)).astype(str).value_counts().to_dict(),
        "word_summary_table": word_summary.fillna("").to_string(),
        "display_table": display_frame[DISPLAY_COLUMNS].fillna("").to_string(),
        "warning": warning,
        "frontend_display_fields": rows[:5],
    }


def _number(value: object, caster):
    try:
        if pd.isna(value):
            return caster(0)
        return caster(value)
    except Exception:
        return caster(0)


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_file_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:80] or "web_demo"


def _uploaded_suffix(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in {".wav", ".mp3", ".m4a", ".flac", ".ogg"} else ".wav"


def _parse_multipart(handler: BaseHTTPRequestHandler, content_type: str) -> tuple[dict[str, str], dict[str, UploadedFile]]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    body = handler.rfile.read(length)
    message_bytes = (
        f"Content-Type: {content_type}\r\n"
        "MIME-Version: 1.0\r\n\r\n"
    ).encode("utf-8") + body
    message = BytesParser(policy=policy.default).parsebytes(message_bytes)
    fields: dict[str, str] = {}
    files: dict[str, UploadedFile] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename:
            files[name] = UploadedFile(filename=filename, content=payload)
        else:
            charset = part.get_content_charset() or "utf-8"
            fields[name] = payload.decode(charset, errors="replace")
    return fields, files


if __name__ == "__main__":
    main()
