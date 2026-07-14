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


APP_VERSION = "DELETION_ONLY_WORD_SUMMARY_DISPLAY_V5"
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
from pronunciation.deletion_detector import build_word_summary, detect_word_deletions  # noqa: E402
from pronunciation.decision import apply_deletion_only_override  # noqa: E402
from pronunciation.final_word_decision import (  # noqa: E402
    merge_word_diagnosis_into_phones,
    run_word_level_diagnosis,
)
from pronunciation.target_words import build_target_word_table, ensure_word_summary_coverage  # noqa: E402


WEB_OUTPUT_DIR = ROOT / "outputs" / "webapp"
DISPLAY_COLUMNS = [
    "word",
    "target_phone",
    "display_decision",
    "display_error",
    "display_align",
    "display_error_type",
    "deletion_trigger_source",
    "missing_word_reason",
    "lexicon_status",
    "g2p_source",
    "g2p_confidence",
    "lexicon_display",
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


def diagnose(audio_path: Path, text: str, utterance_id: str, speaker_id: str, trim_silence: bool = True) -> dict[str, object]:
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
        decision_mode="deletion_only",
        detect_deletion_as_error=False,
        word_summary_output=word_summary_output,
        preprocessed_audio_output=preprocessed_output,
        no_auto_preprocess=False,
        trim_silence=trim_silence,
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
    frame, _ = detect_word_deletions(frame, mode=args.decision_mode)
    out = _final_output(frame, args)
    out = ensure_prediction_coverage(out, target_word_table, g2p)
    word_summary = build_word_summary(out, mode=args.decision_mode)
    word_summary = ensure_word_summary_coverage(target_word_table, word_summary)
    out, word_summary = apply_deletion_only_override(
        out,
        word_summary,
        detect_deletion_as_error=args.detect_deletion_as_error,
    )
    word_summary = run_word_level_diagnosis(out, word_summary)
    word_summary = ensure_word_summary_coverage(target_word_table, word_summary)
    out = merge_word_diagnosis_into_phones(out, word_summary)
    out = ensure_prediction_coverage(out, target_word_table, g2p)
    out.to_csv(output, index=False, encoding="utf-8-sig")
    word_summary.to_csv(word_summary_output, index=False, encoding="utf-8-sig")

    prediction_df = pd.read_csv(output, encoding="utf-8-sig")
    word_summary_df = pd.read_csv(word_summary_output, encoding="utf-8-sig")
    display_frame = apply_word_summary_display(prediction_df, word_summary_df, args.decision_mode)
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
                "deletion_trigger_source",
            ]
        ].to_string(),
        flush=True,
    )
    rows = display_frame[DISPLAY_COLUMNS].fillna("").to_dict(orient="records")
    counts = display_frame["display_decision"].value_counts().to_dict()
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
        "n_correct": int(counts.get("正确", 0)),
        "n_acceptable_accent": 0,
        "n_true_error": int(counts.get("漏读", 0)),
        "n_uncertain_review": int(counts.get("疑似漏读/需复核", 0) + counts.get("需复核", 0)),
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
            payload = diagnose(tmp_path, text, utterance_id, speaker_id, trim_silence=trim_silence)
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
        word_error = display_df["word_error_type"].fillna("").astype(str)
        phone_error = display_df["error_type"].fillna("").astype(str)
        phone_alignment = display_df["alignment_quality"].fillna("").astype(str).str.strip().str.lower()
        word_alignment = display_df["word_alignment_quality"].fillna("").astype(str).str.strip().str.lower()
        possible_missing = display_df["possible_missing_word"].map(_boolish)
        deletion = word_error.eq("deletion")
        possible = word_error.eq("possible_deletion")
        g2p_issue = word_error.eq("g2p_issue") | display_df.get(
            "lexicon_status", pd.Series("", index=display_df.index)
        ).fillna("").astype(str).eq("failed")
        alignment_issue = (
            word_error.eq("alignment_issue")
            | phone_error.eq("alignment_issue")
            | phone_alignment.isin({"bad", "failed", "alignment_failed"})
            | word_alignment.isin({"bad", "failed", "alignment_failed"})
        )

        display_df.loc[alignment_issue, "display_decision"] = "需复核"
        display_df.loc[alignment_issue, "display_error"] = "对齐失败"
        display_df.loc[alignment_issue, "display_align"] = "bad"
        display_df.loc[alignment_issue, "display_error_type"] = "alignment_issue"

        display_df.loc[possible_missing, "display_decision"] = "疑似漏读/需复核"
        display_df.loc[possible_missing, "display_error"] = "疑似漏读"
        display_df.loc[possible_missing, "display_align"] = "suspect"
        display_df.loc[possible_missing, "display_error_type"] = word_error[possible_missing].where(
            word_error[possible_missing].ne(""),
            "possible_deletion",
        )

        display_df.loc[deletion, "display_decision"] = "漏读"
        display_df.loc[deletion, "display_error"] = "漏读"
        display_df.loc[deletion, "display_align"] = "suspect"
        display_df.loc[deletion, "display_error_type"] = "deletion"

        display_df.loc[possible, "display_decision"] = "疑似漏读/需复核"
        display_df.loc[possible, "display_error"] = "疑似漏读"
        display_df.loc[possible, "display_align"] = "suspect"
        display_df.loc[possible, "display_error_type"] = "possible_deletion"

        display_df.loc[g2p_issue, "display_decision"] = "需复核"
        display_df.loc[g2p_issue, "display_error"] = "词典缺失"
        display_df.loc[g2p_issue, "display_align"] = "bad"
        display_df.loc[g2p_issue, "display_error_type"] = "g2p_issue"

    return display_df


def deletion_only_display_fields(row: dict[str, object]) -> dict[str, str]:
    if row.get("display_error") or row.get("display_decision") or row.get("display_align"):
        display_error_type = str(row.get("display_error_type", ""))
        display_align = str(row.get("display_align", "")) or str(row.get("alignment_quality", ""))
        if display_error_type == "g2p_issue":
            return {"error_display": "词典缺失", "decision_display": "需复核", "align_display": "bad"}
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
        error_display = "词典缺失"
        decision_display = "需复核"
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
