from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
import threading
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from phoneme_assessment.phones import phone_group


DEFAULT_PHONE_CTC_MODEL = "mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme"
DEFAULT_REFERENCE_PHONE_CTC_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
DEFAULT_THREE_STATE_MODEL = Path(__file__).resolve().parents[2] / "models/phone_three_state_v5.joblib"
REFERENCE_DELETION_MARGIN_THRESHOLD = 3.0
PHONE_PRESENCE_MAX_DELETION_MARGIN = 3.0
PHONE_PRESENCE_MIN_DURATION_MS = 70.0
PHONE_STATES = ("correct", "mispronounced", "deleted")
STATE_ZH = {"correct": "读对", "mispronounced": "读错", "deleted": "漏读", "unavailable": "单词暂未收录"}

# These narrow pairs cover common dictionary/allophonic alternatives observed
# in read English. Mandarin-specific confusions such as IH/IY remain errors.
DEPLOYMENT_EQUIVALENT_PHONE_PAIRS = {
    ("AA", "AO"),
    ("AO", "AA"),
    ("AH", "IH"),
    ("IH", "AH"),
    ("UH", "UW"),
    ("UW", "UH"),
}

_MODEL_LOCK = threading.Lock()


# The L2-ARCTIC checkpoint emits character-level IPA. Multi-character phones
# such as diphthongs and affricates are represented by multiple CTC labels.
ARPABET_TO_IPA: dict[str, tuple[str, ...]] = {
    "AA": ("ɑ",),
    "AE": ("æ",),
    "AH": ("ʌ",),
    "AO": ("ɔ",),
    "AW": ("a", "ʊ"),
    "AY": ("a", "ɪ"),
    "EH": ("ɛ",),
    "ER": ("ɚ",),
    "EY": ("e", "ɪ"),
    "IH": ("ɪ",),
    "IY": ("i",),
    "OW": ("o", "ʊ"),
    "OY": ("ɔ", "ɪ"),
    "UH": ("ʊ",),
    "UW": ("u",),
    "B": ("b",),
    "CH": ("t", "͡", "ʃ"),
    "D": ("d",),
    "DH": ("ð",),
    "F": ("f",),
    "G": ("ɡ",),
    "HH": ("h",),
    "JH": ("d", "͡", "ʒ"),
    "K": ("k",),
    "L": ("l",),
    "M": ("m",),
    "N": ("n",),
    "NG": ("ŋ",),
    "P": ("p",),
    "R": ("ɹ",),
    "S": ("s",),
    "SH": ("ʃ",),
    "T": ("t",),
    "TH": ("θ",),
    "V": ("v",),
    "W": ("w",),
    "Y": ("j",),
    "Z": ("z",),
    "ZH": ("ʒ",),
}

VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
STOPS = {"P", "B", "T", "D", "K", "G"}
FRICATIVES = {"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"}
AFFRICATES = {"CH", "JH"}
NASALS = {"M", "N", "NG"}
LIQUIDS_GLIDES = {"L", "R", "W", "Y"}

MANDARIN_CONFUSIONS: dict[str, set[str]] = {
    "TH": {"S", "T", "F", "D"},
    "DH": {"D", "Z", "T"},
    "R": {"L", "W"},
    "L": {"R", "W"},
    "V": {"W", "F"},
    "W": {"V"},
    "IH": {"IY"},
    "IY": {"IH"},
    "AE": {"EH"},
    "EH": {"AE"},
    "P": {"B"},
    "B": {"P"},
    "T": {"D"},
    "D": {"T"},
    "K": {"G"},
    "G": {"K"},
    "F": {"V"},
    "S": {"Z", "TH"},
    "Z": {"S", "DH"},
    "SH": {"ZH"},
    "ZH": {"SH"},
    "CH": {"JH", "SH"},
    "JH": {"CH", "ZH"},
}


def normalize_arpabet(phone: object) -> str:
    text = "" if phone is None or pd.isna(phone) else str(phone).strip().upper()
    return text.rstrip("0123456789")


def phone_equivalence_guard(
    frame: pd.DataFrame,
    *,
    recognized_column: str = "recognized_phone",
) -> pd.Series:
    """Return rows where substitution evidence is not strong enough to call an error."""
    target = frame.get("target_phone", pd.Series("", index=frame.index)).map(normalize_arpabet)
    recognized = frame.get(recognized_column, pd.Series("", index=frame.index)).map(normalize_arpabet)
    pairs = pd.Series(list(zip(target, recognized)), index=frame.index)
    word = frame.get("word", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper()
    of_v_devoicing = word.eq("OF") & target.eq("V") & recognized.eq("F")
    return recognized.ne("") & (
        recognized.eq(target)
        | pairs.isin(DEPLOYMENT_EQUIVALENT_PHONE_PAIRS)
        | of_v_devoicing
    )


def dual_phone_presence_guard(frame: pd.DataFrame) -> pd.Series:
    """Reject deletion when both recognizers hear a sufficiently long target phone."""
    primary_match = frame.get(
        "primary_target_match",
        pd.Series(False, index=frame.index),
    ).fillna(False).astype(bool)
    reference_match = frame.get(
        "reference_target_match",
        pd.Series(False, index=frame.index),
    ).fillna(False).astype(bool)
    primary_margin = pd.to_numeric(
        frame.get("ctc_deletion_margin", pd.Series(float("nan"), index=frame.index)),
        errors="coerce",
    )
    reference_margin = pd.to_numeric(
        frame.get("reference_ctc_deletion_margin", pd.Series(float("nan"), index=frame.index)),
        errors="coerce",
    )
    duration = pd.to_numeric(
        frame.get("duration_ms", pd.Series(float("nan"), index=frame.index)),
        errors="coerce",
    )
    return (
        primary_match
        & reference_match
        & primary_margin.le(PHONE_PRESENCE_MAX_DELETION_MARGIN)
        & reference_margin.le(PHONE_PRESENCE_MAX_DELETION_MARGIN)
        & duration.ge(PHONE_PRESENCE_MIN_DURATION_MS)
    )


def prefix_reference_phone_evidence(evidence: pd.DataFrame) -> pd.DataFrame:
    """Prefix a second acoustic model's columns while preserving merge keys."""
    keys = {"word_index", "phone_index"}
    return evidence.rename(
        columns={column: f"reference_{column}" for column in evidence.columns if column not in keys}
    )


def add_phone_model_consensus_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add numeric agreement features shared by training and inference."""
    out = frame.copy()
    target = out.get("target_phone", pd.Series("", index=out.index)).map(normalize_arpabet)
    primary = out.get("recognized_phone", pd.Series("", index=out.index)).map(normalize_arpabet)
    reference = out.get("reference_recognized_phone", pd.Series("", index=out.index)).map(normalize_arpabet)
    primary_available = primary.ne("")
    reference_available = reference.ne("")
    out["primary_recognized_available"] = primary_available.astype(float)
    out["reference_recognized_available"] = reference_available.astype(float)
    out["primary_target_match"] = (primary_available & primary.eq(target)).astype(float)
    out["reference_target_match"] = (reference_available & reference.eq(target)).astype(float)
    out["phone_models_recognized_same"] = (
        primary_available & reference_available & primary.eq(reference)
    ).astype(float)

    primary_deletion = pd.to_numeric(
        out.get("ctc_deletion_margin", pd.Series(float("nan"), index=out.index)),
        errors="coerce",
    )
    reference_deletion = pd.to_numeric(
        out.get("reference_ctc_deletion_margin", pd.Series(float("nan"), index=out.index)),
        errors="coerce",
    )
    primary_substitution = pd.to_numeric(
        out.get("ctc_substitution_margin", pd.Series(float("nan"), index=out.index)),
        errors="coerce",
    )
    reference_substitution = pd.to_numeric(
        out.get("reference_ctc_substitution_margin", pd.Series(float("nan"), index=out.index)),
        errors="coerce",
    )
    out["dual_deletion_margin_min"] = pd.concat(
        [primary_deletion, reference_deletion], axis=1
    ).min(axis=1, skipna=True)
    out["dual_deletion_margin_max"] = pd.concat(
        [primary_deletion, reference_deletion], axis=1
    ).max(axis=1, skipna=True)
    out["dual_substitution_margin_min"] = pd.concat(
        [primary_substitution, reference_substitution], axis=1
    ).min(axis=1, skipna=True)
    out["dual_substitution_margin_max"] = pd.concat(
        [primary_substitution, reference_substitution], axis=1
    ).max(axis=1, skipna=True)
    return out


def arpabet_to_ipa_tokens(phone: object) -> tuple[str, ...]:
    return ARPABET_TO_IPA.get(normalize_arpabet(phone), ())


def substitution_candidates(phone: object) -> list[str]:
    target = normalize_arpabet(phone)
    if target in VOWELS:
        family = VOWELS
    elif target in STOPS:
        family = STOPS
    elif target in FRICATIVES:
        family = FRICATIVES
    elif target in AFFRICATES:
        family = AFFRICATES | FRICATIVES
    elif target in NASALS:
        family = NASALS
    elif target in LIQUIDS_GLIDES:
        family = LIQUIDS_GLIDES
    else:
        family = set(ARPABET_TO_IPA)
    values = (set(family) | MANDARIN_CONFUSIONS.get(target, set())) - {target}
    return sorted(value for value in values if value in ARPABET_TO_IPA)


def score_audio_phones_ctc(
    audio_path: Path,
    phone_frame: pd.DataFrame,
    *,
    model_id: str = DEFAULT_PHONE_CTC_MODEL,
    local_files_only: bool = True,
) -> pd.DataFrame:
    """Compute alignment-free deletion/substitution evidence for target phones."""
    base = phone_frame.copy().reset_index(drop=True)
    columns = _evidence_columns()
    if base.empty:
        return pd.DataFrame(columns=columns)
    try:
        processor, model = _load_model(model_id, local_files_only)
        samples = _read_audio_16k(Path(audio_path))
        inputs = processor(samples, sampling_rate=16000, return_tensors="pt")
        import torch

        parameter = next(model.parameters())
        input_values = inputs.input_values.to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        with _MODEL_LOCK, torch.inference_mode():
            logits = model(input_values).logits.squeeze(0).float().cpu()
        log_probs = logits.log_softmax(dim=-1)
        vocab = processor.tokenizer.get_vocab()
        blank_id = int(model.config.pad_token_id)
        labels, phone_spans, available = build_ctc_target_sequence(base, vocab)
        if not labels:
            raise ValueError("no target phones can be represented by the CTC vocabulary")

        sequences: list[list[int]] = [labels]
        sequence_meta: list[tuple[int, str, str]] = [(-1, "reference", "")]
        for index, ((start, end), is_available) in enumerate(zip(phone_spans, available)):
            if not is_available:
                continue
            sequences.append(labels[:start] + labels[end:])
            sequence_meta.append((index, "deletion", ""))
            for candidate in substitution_candidates(base.iloc[index].get("target_phone")):
                candidate_ids = _tokens_to_ids(ARPABET_TO_IPA[candidate], vocab)
                if candidate_ids:
                    sequences.append(labels[:start] + candidate_ids + labels[end:])
                    sequence_meta.append((index, "substitution", candidate))

        log_likelihoods = ctc_sequence_log_probabilities(log_probs, sequences, blank_id=blank_id)
        reference_log_probability = float(log_likelihoods[0])
        variants: dict[int, list[tuple[str, str, float]]] = {index: [] for index in range(len(base))}
        for meta, value in zip(sequence_meta[1:], log_likelihoods[1:]):
            index, operation, candidate = meta
            variants[index].append((operation, candidate, float(value)))

        aligned_states = ctc_viterbi_states(log_probs.numpy(), labels, blank_id=blank_id)
        frame_seconds = (len(samples) / 16000.0) / max(int(logits.shape[0]), 1)
        greedy_ipa = processor.batch_decode(logits.argmax(dim=-1).unsqueeze(0))[0]
        rows: list[dict[str, object]] = []
        for index, row in base.iterrows():
            target = normalize_arpabet(row.get("target_phone"))
            start, end = phone_spans[index]
            token_count = max(1, end - start)
            deletion_values = [value for op, _, value in variants[index] if op == "deletion"]
            substitutions = [(candidate, value) for op, candidate, value in variants[index] if op == "substitution"]
            deletion_log_probability = deletion_values[0] if deletion_values else float("nan")
            best_phone, best_sub_log_probability = max(substitutions, key=lambda item: item[1]) if substitutions else ("", float("nan"))
            deletion_margin = (
                (deletion_log_probability - reference_log_probability) / token_count
                if math.isfinite(deletion_log_probability)
                else float("nan")
            )
            substitution_margin = (
                (best_sub_log_probability - reference_log_probability) / token_count
                if math.isfinite(best_sub_log_probability)
                else float("nan")
            )
            frame_indices = _frames_for_phone(aligned_states, start, end)
            target_span_score, competing_span_score, span_best_phone = _span_logit_scores(
                logits.numpy(), frame_indices, target, vocab
            )
            logit_margin = target_span_score - competing_span_score
            target_log_probability = _target_path_log_probability(
                log_probs.numpy(), aligned_states, start, end, labels
            )
            start_frame = int(frame_indices.min()) if len(frame_indices) else 0
            end_frame = int(frame_indices.max()) + 1 if len(frame_indices) else start_frame
            recognized = best_phone if substitution_margin > 0 else (span_best_phone if logit_margin < 0 else target)
            rows.append(
                {
                    "word": row.get("word", ""),
                    "word_index": row.get("word_index", 0),
                    "phone_index": row.get("phone_index", index),
                    "target_phone": target,
                    "recognized_phone": recognized,
                    "ctc_reference_log_probability": reference_log_probability,
                    "ctc_deletion_log_probability": deletion_log_probability,
                    "ctc_best_substitution_log_probability": best_sub_log_probability,
                    "ctc_deletion_margin": deletion_margin,
                    "ctc_substitution_margin": substitution_margin,
                    "ctc_target_logit_score": target_span_score,
                    "ctc_competing_logit_score": competing_span_score,
                    "ctc_logit_margin": logit_margin,
                    "ctc_target_path_log_probability": target_log_probability,
                    "ctc_start_ms": round(start_frame * frame_seconds * 1000.0, 3),
                    "ctc_end_ms": round(end_frame * frame_seconds * 1000.0, 3),
                    "ctc_duration_ms": round((end_frame - start_frame) * frame_seconds * 1000.0, 3),
                    "ctc_greedy_ipa": greedy_ipa,
                    "ctc_phone_model_available": bool(available[index]),
                    "ctc_phone_model": model_id,
                    "ctc_phone_error": "" if available[index] else f"unsupported_target_phone:{target}",
                }
            )
        return pd.DataFrame(rows, columns=columns)
    except Exception as error:
        return _unavailable_evidence(base, model_id, error)


def build_ctc_target_sequence(
    phone_frame: pd.DataFrame,
    vocab: dict[str, int],
) -> tuple[list[int], list[tuple[int, int]], list[bool]]:
    labels: list[int] = []
    spans: list[tuple[int, int]] = []
    available: list[bool] = []
    previous_word: object = None
    delimiter_id = vocab.get(" ")
    for position, (_, row) in enumerate(phone_frame.iterrows()):
        word_key = row.get("word_index")
        if word_key is None or pd.isna(word_key) or str(word_key).strip() == "":
            word_key = row.get("word", position)
        if position and word_key != previous_word and delimiter_id is not None:
            labels.append(int(delimiter_id))
        tokens = arpabet_to_ipa_tokens(row.get("target_phone"))
        ids = _tokens_to_ids(tokens, vocab)
        start = len(labels)
        labels.extend(ids)
        spans.append((start, len(labels)))
        available.append(bool(ids) and len(ids) == len(tokens))
        previous_word = word_key
    return labels, spans, available


def ctc_sequence_log_probabilities(
    log_probs: object,
    sequences: list[list[int]],
    *,
    blank_id: int,
    batch_size: int = 128,
) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    values = torch.as_tensor(log_probs, dtype=torch.float32)
    if values.ndim != 2:
        raise ValueError("log_probs must have shape [time, vocabulary]")
    results: list[np.ndarray] = []
    for offset in range(0, len(sequences), batch_size):
        batch = sequences[offset : offset + batch_size]
        lengths = torch.tensor([len(sequence) for sequence in batch], dtype=torch.long)
        targets = torch.tensor([token for sequence in batch for token in sequence], dtype=torch.long)
        emissions = values.unsqueeze(1).expand(-1, len(batch), -1)
        input_lengths = torch.full((len(batch),), values.shape[0], dtype=torch.long)
        losses = functional.ctc_loss(
            emissions,
            targets,
            input_lengths,
            lengths,
            blank=int(blank_id),
            reduction="none",
            zero_infinity=True,
        )
        results.append((-losses).cpu().numpy())
    return np.concatenate(results) if results else np.empty(0, dtype=np.float32)


def ctc_viterbi_states(log_probs: np.ndarray, labels: list[int], *, blank_id: int) -> np.ndarray:
    """Return the best extended-CTC state index for every acoustic frame."""
    emissions = np.asarray(log_probs, dtype=np.float64)
    extended = [blank_id]
    for label in labels:
        extended.extend([int(label), blank_id])
    time_steps, _ = emissions.shape
    states = len(extended)
    scores = np.full((time_steps, states), -np.inf, dtype=np.float64)
    back = np.full((time_steps, states), -1, dtype=np.int32)
    scores[0, 0] = emissions[0, blank_id]
    if states > 1:
        scores[0, 1] = emissions[0, extended[1]]
    for time in range(1, time_steps):
        for state, label in enumerate(extended):
            candidates = [(scores[time - 1, state], state)]
            if state > 0:
                candidates.append((scores[time - 1, state - 1], state - 1))
            if state > 1 and label != blank_id and label != extended[state - 2]:
                candidates.append((scores[time - 1, state - 2], state - 2))
            best_score, previous = max(candidates, key=lambda item: item[0])
            scores[time, state] = best_score + emissions[time, label]
            back[time, state] = previous
    final_candidates = [states - 1]
    if states > 1:
        final_candidates.append(states - 2)
    state = max(final_candidates, key=lambda value: scores[-1, value])
    path = np.empty(time_steps, dtype=np.int32)
    for time in range(time_steps - 1, -1, -1):
        path[time] = state
        if time:
            state = int(back[time, state])
            if state < 0:
                state = 0
    return path


def apply_phone_three_state_model(
    phone_frame: pd.DataFrame,
    evidence: pd.DataFrame,
    *,
    classifier_path: Path = DEFAULT_THREE_STATE_MODEL,
    reference_evidence: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = phone_frame.copy()
    keys = [key for key in ("phone_index", "word_index") if key in out.columns and key in evidence.columns]
    if not keys:
        raise KeyError("phone_frame and evidence must share phone_index or word_index")
    replace_columns = [column for column in evidence.columns if column in out.columns and column not in keys]
    out = out.drop(columns=replace_columns)
    evidence_columns = list(evidence.columns)
    out = out.merge(evidence[evidence_columns].drop_duplicates(keys, keep="last"), on=keys, how="left")
    if reference_evidence is not None and not reference_evidence.empty:
        reference = prefix_reference_phone_evidence(reference_evidence)
        reference_keys = [key for key in keys if key in reference.columns]
        reference_columns = [column for column in reference.columns if column not in out.columns or column in reference_keys]
        out = out.merge(
            reference[reference_columns].drop_duplicates(reference_keys, keep="last"),
            on=reference_keys,
            how="left",
        )
    if "phone_group" not in out.columns:
        out["phone_group"] = out["target_phone"].map(phone_group)
    else:
        missing_group = out["phone_group"].fillna("").astype(str).str.strip().eq("")
        out.loc[missing_group, "phone_group"] = out.loc[missing_group, "target_phone"].map(phone_group)
    out = add_phone_model_consensus_features(out)

    artifact = _load_classifier(Path(classifier_path)) if Path(classifier_path).is_file() else None
    probabilities = _heuristic_probabilities(out)
    source = "ctc_hypothesis_rules_v1"
    thresholds = {"deleted": 0.62, "mispronounced": 0.62}
    use_reference_hard_gates = True
    if artifact is not None:
        feature_frame = _classifier_feature_frame(out, artifact)
        model = artifact["pipeline"]
        raw = model.predict_proba(feature_frame)
        classes = [str(value) for value in model.classes_]
        probabilities = pd.DataFrame(0.0, index=out.index, columns=list(PHONE_STATES))
        for index, label in enumerate(classes):
            if label in probabilities.columns:
                probabilities[label] = raw[:, index]
        thresholds.update(artifact.get("thresholds", {}))
        source = str(artifact.get("name", "l2_arctic_ctc_three_state_v1"))
        use_reference_hard_gates = bool(artifact.get("use_reference_hard_gates", True))

    primary_guard = phone_equivalence_guard(out)
    reference_guard = phone_equivalence_guard(out, recognized_column="reference_recognized_phone")
    presence_guard = dual_phone_presence_guard(out)
    reference_available = out.get(
        "reference_ctc_phone_model_available",
        pd.Series(False, index=out.index),
    ).fillna(False).astype(bool)
    reference_deletion_margin = pd.to_numeric(
        out.get("reference_ctc_deletion_margin", pd.Series(float("nan"), index=out.index)),
        errors="coerce",
    )
    deletion_rejected_by_reference = presence_guard | (
        reference_available & reference_deletion_margin.lt(REFERENCE_DELETION_MARGIN_THRESHOLD)
        if use_reference_hard_gates
        else pd.Series(False, index=out.index)
    )
    rejected_deletion_probability = probabilities.loc[deletion_rejected_by_reference, "deleted"].copy()
    probabilities.loc[deletion_rejected_by_reference, "deleted"] = 0.0
    probabilities.loc[deletion_rejected_by_reference, "correct"] = (
        probabilities.loc[deletion_rejected_by_reference, "correct"] + rejected_deletion_probability
    ).clip(upper=1.0)

    equivalence_guard = primary_guard | reference_guard if use_reference_hard_gates else primary_guard
    protected_probability = probabilities.loc[equivalence_guard, "mispronounced"].copy()
    probabilities.loc[equivalence_guard, "mispronounced"] = 0.0
    probabilities.loc[equivalence_guard, "correct"] = (
        probabilities.loc[equivalence_guard, "correct"] + protected_probability
    ).clip(upper=1.0)

    available = out.get("ctc_phone_model_available", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    states: list[str] = []
    for index in out.index:
        if not available.loc[index]:
            states.append("unavailable")
        elif probabilities.loc[index, "deleted"] >= float(thresholds["deleted"]):
            states.append("deleted")
        elif probabilities.loc[index, "mispronounced"] >= float(thresholds["mispronounced"]):
            states.append("mispronounced")
        else:
            states.append("correct")

    out["phone_probability_correct"] = probabilities["correct"].round(6)
    out["phone_probability_mispronounced"] = probabilities["mispronounced"].round(6)
    out["phone_probability_deleted"] = probabilities["deleted"].round(6)
    out["phone_state"] = states
    out["phone_state_zh"] = out["phone_state"].map(STATE_ZH)
    out["phone_state_source"] = source
    out["phone_equivalence_guard"] = equivalence_guard
    out["primary_phone_equivalence_guard"] = primary_guard
    out["reference_phone_equivalence_guard"] = reference_guard
    out["dual_phone_presence_guard"] = presence_guard
    out["reference_deletion_supported"] = reference_available & reference_deletion_margin.ge(
        REFERENCE_DELETION_MARGIN_THRESHOLD
    )
    out["phone_state_confidence"] = [
        0.0 if state == "unavailable" else float(probabilities.loc[index, state])
        for index, state in zip(out.index, states)
    ]
    unavailable = out["phone_state"].eq("unavailable")
    out.loc[unavailable, "phone_probability_correct"] = 0.5
    out.loc[unavailable, "phone_probability_mispronounced"] = 0.5
    out.loc[unavailable, "phone_probability_deleted"] = 0.0
    out["phone_error_probability"] = (
        out["phone_probability_mispronounced"] + out["phone_probability_deleted"]
    ).clip(0.0, 1.0).round(6)
    out["phone_error_percent"] = (out["phone_error_probability"] * 100.0).round(2)
    out["phone_decision"] = out["phone_state"].map(
        {"correct": "correct", "mispronounced": "true_error", "deleted": "true_error", "unavailable": "uncertain_review"}
    )
    out["phone_error_type"] = out["phone_state"].map(
        {"correct": "", "mispronounced": "mispronunciation", "deleted": "deletion", "unavailable": "g2p_issue"}
    )
    g2p_failed = (
        out.get("g2p_status", pd.Series("success", index=out.index)).fillna("failed").astype(str).eq("failed")
        | out.get("target_phone", pd.Series("", index=out.index)).fillna("").astype(str).eq("<UNK>")
    )
    bad_alignment = out.get(
        "alignment_quality", pd.Series("", index=out.index)
    ).fillna("").astype(str).str.lower().isin({"bad", "failed", "alignment_failed"})
    out.loc[unavailable & ~g2p_failed & bad_alignment, "phone_error_type"] = "alignment_issue"
    out.loc[unavailable & ~g2p_failed & ~bad_alignment, "phone_error_type"] = "model_unavailable"
    out.loc[unavailable & ~g2p_failed, "phone_state_zh"] = "暂无法判断"
    out["phone_confidence"] = out["phone_state_confidence"].round(6)
    out["phone_score_source"] = source
    out["decision"] = out["phone_decision"]
    out["error_type"] = out["phone_error_type"]
    out["confidence"] = out["phone_confidence"]
    out["evidence_summary"] = out.apply(_evidence_summary, axis=1)
    out["review_reason"] = out["evidence_summary"]
    return out


def force_confirmed_word_deletions(phone_frame: pd.DataFrame, word_summary: pd.DataFrame) -> pd.DataFrame:
    """Give confirmed word deletion priority over phone substitution evidence."""
    out = phone_frame.copy()
    if out.empty or word_summary.empty or "word_index" not in out.columns:
        return out
    summary = word_summary.copy()
    deletion = summary.get("final_error_type", summary.get("error_type", pd.Series("", index=summary.index))).astype(str).eq("deletion")
    deletion |= summary.get("deletion_decision", pd.Series("", index=summary.index)).astype(str).eq("deletion")
    deleted_words = set(summary.loc[deletion, "word_index"].astype(str))
    mask = out["word_index"].astype(str).isin(deleted_words)
    if not mask.any():
        return out
    out.loc[mask, "phone_state"] = "deleted"
    out.loc[mask, "phone_state_zh"] = STATE_ZH["deleted"]
    out.loc[mask, "phone_probability_deleted"] = out.loc[mask, "phone_probability_deleted"].clip(lower=0.95)
    out.loc[mask, "phone_probability_correct"] = 0.0
    out.loc[mask, "phone_probability_mispronounced"] = 0.0
    out.loc[mask, "phone_error_probability"] = 1.0
    out.loc[mask, "phone_error_percent"] = 100.0
    out.loc[mask, "phone_decision"] = "true_error"
    out.loc[mask, "phone_error_type"] = "deletion"
    out.loc[mask, "phone_state_confidence"] = out.loc[mask, "phone_probability_deleted"]
    out.loc[mask, "phone_confidence"] = out.loc[mask, "phone_state_confidence"]
    out.loc[mask, "decision"] = "true_error"
    out.loc[mask, "error_type"] = "deletion"
    out.loc[mask, "evidence_summary"] = "Confirmed word deletion overrides phone-level substitution evidence."
    out.loc[mask, "review_reason"] = out.loc[mask, "evidence_summary"]
    return out


def summarize_three_state_phones(phone_frame: pd.DataFrame) -> pd.DataFrame:
    if phone_frame.empty or "word_index" not in phone_frame.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for word_index, group in phone_frame.groupby("word_index", sort=False, dropna=False):
        states = group.get("phone_state", pd.Series("unavailable", index=group.index)).astype(str)
        if states.eq("deleted").all():
            word_state, error_type = "deleted", "deletion"
        elif states.eq("mispronounced").any():
            word_state, error_type = "mispronounced", "mispronunciation"
        elif states.eq("deleted").any():
            word_state, error_type = "mispronounced", "partial_phone_deletion"
        elif states.eq("unavailable").any():
            word_state, error_type = "unavailable", "g2p_issue"
        else:
            word_state, error_type = "correct", ""
        rows.append(
            {
                "word_index": word_index,
                "word": _first(group, "word"),
                "phone_count": int(len(group)),
                "word_decision": word_state,
                "error_type": error_type,
                "alignment_quality": _first(group, "alignment_quality"),
                "num_phone_correct": int(states.eq("correct").sum()),
                "num_phone_mispronounced": int(states.eq("mispronounced").sum()),
                "num_phone_deleted": int(states.eq("deleted").sum()),
                "num_phone_unavailable": int(states.eq("unavailable").sum()),
            }
        )
    return pd.DataFrame(rows)


def _heuristic_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    deletion_margin = pd.to_numeric(frame.get("ctc_deletion_margin"), errors="coerce").fillna(-10.0)
    substitution_margin = pd.to_numeric(frame.get("ctc_substitution_margin"), errors="coerce").fillna(-10.0)
    logit_margin = pd.to_numeric(frame.get("ctc_logit_margin"), errors="coerce").fillna(0.0)
    deletion = deletion_margin.map(lambda value: _sigmoid((value - 1.0) / 1.5))
    wrong = pd.concat(
        [substitution_margin.map(lambda value: _sigmoid((value - 0.75) / 1.5)), (-logit_margin).map(lambda value: _sigmoid((value - 1.0) / 2.0))],
        axis=1,
    ).mean(axis=1)
    wrong = wrong * (1.0 - deletion)
    correct = (1.0 - deletion - wrong).clip(lower=0.0)
    total = (correct + wrong + deletion).replace(0.0, 1.0)
    return pd.DataFrame(
        {
            "correct": correct / total,
            "mispronounced": wrong / total,
            "deleted": deletion / total,
        },
        index=frame.index,
    )


def _classifier_feature_frame(frame: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    columns = list(artifact.get("feature_columns", []))
    categorical = set(artifact.get("categorical_features", []))
    features = pd.DataFrame(index=frame.index)
    for column in columns:
        if column in categorical:
            features[column] = frame.get(column, pd.Series("", index=frame.index)).fillna("").astype(str)
        else:
            features[column] = pd.to_numeric(frame.get(column, pd.Series(float("nan"), index=frame.index)), errors="coerce")
    return features


@lru_cache(maxsize=2)
def _load_classifier(path: Path) -> dict:
    return joblib.load(path)


@lru_cache(maxsize=2)
def _load_model(model_id: str, local_files_only: bool):
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        do_phonemize=False,
    )
    model = AutoModelForCTC.from_pretrained(model_id, local_files_only=local_files_only)
    if torch.cuda.is_available():
        model = model.to(device="cuda", dtype=torch.float16)
    else:
        model = model.to(device="cpu")
    model.eval()
    return processor, model


def _tokens_to_ids(tokens: Iterable[str], vocab: dict[str, int]) -> list[int]:
    values = list(tokens)
    if any(token not in vocab for token in values):
        return []
    return [int(vocab[token]) for token in values]


def _frames_for_phone(states: np.ndarray, start: int, end: int) -> np.ndarray:
    phone_states = [2 * label_index + 1 for label_index in range(start, end)]
    return np.flatnonzero(np.isin(states, phone_states))


def _target_path_log_probability(
    log_probs: np.ndarray,
    states: np.ndarray,
    start: int,
    end: int,
    labels: list[int],
) -> float:
    values: list[float] = []
    for label_index in range(start, end):
        frames = np.flatnonzero(states == 2 * label_index + 1)
        if len(frames):
            values.extend(log_probs[frames, labels[label_index]].tolist())
    return float(np.mean(values)) if values else float("nan")


def _span_logit_scores(
    logits: np.ndarray,
    frames: np.ndarray,
    target: str,
    vocab: dict[str, int],
) -> tuple[float, float, str]:
    if not len(frames):
        return float("nan"), float("nan"), ""
    target_ids = _tokens_to_ids(ARPABET_TO_IPA.get(target, ()), vocab)
    target_score = _candidate_span_score(logits, frames, target_ids)
    candidates = []
    for candidate in substitution_candidates(target):
        ids = _tokens_to_ids(ARPABET_TO_IPA[candidate], vocab)
        if ids:
            candidates.append((candidate, _candidate_span_score(logits, frames, ids)))
    best_phone, best_score = max(candidates, key=lambda item: item[1]) if candidates else ("", float("nan"))
    return target_score, best_score, best_phone


def _candidate_span_score(logits: np.ndarray, frames: np.ndarray, token_ids: list[int]) -> float:
    if not token_ids:
        return float("nan")
    return float(np.mean([np.max(logits[frames, token_id]) for token_id in token_ids]))


def _read_audio_16k(path: Path) -> np.ndarray:
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    rate, samples = wavfile.read(path)
    values = np.asarray(samples)
    if values.ndim == 2:
        values = values.mean(axis=1)
    if np.issubdtype(values.dtype, np.integer):
        scale = float(max(abs(np.iinfo(values.dtype).min), np.iinfo(values.dtype).max))
        values = values.astype(np.float32) / scale
    else:
        values = values.astype(np.float32)
    if int(rate) != 16000:
        divisor = math.gcd(int(rate), 16000)
        values = resample_poly(values, 16000 // divisor, int(rate) // divisor).astype(np.float32)
    return values


def _unavailable_evidence(frame: pd.DataFrame, model_id: str, error: Exception) -> pd.DataFrame:
    rows = []
    message = f"{type(error).__name__}: {error}"
    for index, row in frame.iterrows():
        values = {column: float("nan") for column in _numeric_evidence_columns()}
        values.update(
            {
                "word": row.get("word", ""),
                "word_index": row.get("word_index", 0),
                "phone_index": row.get("phone_index", index),
                "target_phone": normalize_arpabet(row.get("target_phone")),
                "recognized_phone": "",
                "ctc_greedy_ipa": "",
                "ctc_phone_model_available": False,
                "ctc_phone_model": model_id,
                "ctc_phone_error": message,
            }
        )
        rows.append(values)
    return pd.DataFrame(rows, columns=_evidence_columns())


def _numeric_evidence_columns() -> list[str]:
    return [
        "ctc_reference_log_probability",
        "ctc_deletion_log_probability",
        "ctc_best_substitution_log_probability",
        "ctc_deletion_margin",
        "ctc_substitution_margin",
        "ctc_target_logit_score",
        "ctc_competing_logit_score",
        "ctc_logit_margin",
        "ctc_target_path_log_probability",
        "ctc_start_ms",
        "ctc_end_ms",
        "ctc_duration_ms",
    ]


def _evidence_columns() -> list[str]:
    return [
        "word",
        "word_index",
        "phone_index",
        "target_phone",
        "recognized_phone",
        *_numeric_evidence_columns(),
        "ctc_greedy_ipa",
        "ctc_phone_model_available",
        "ctc_phone_model",
        "ctc_phone_error",
    ]


def _evidence_summary(row: pd.Series) -> str:
    if str(row.get("phone_state")) == "unavailable":
        if str(row.get("phone_error_type")) == "alignment_issue":
            return f"Alignment failed; CTC phone model unavailable: {row.get('ctc_phone_error', '')}"
        return str(row.get("ctc_phone_error", "CTC phone model unavailable"))
    return (
        f"state={row.get('phone_state')}; recognized={row.get('recognized_phone', '')}; "
        f"reference_recognized={row.get('reference_recognized_phone', '')}; "
        f"deletion_margin={_fmt(row.get('ctc_deletion_margin'))}; "
        f"reference_deletion_margin={_fmt(row.get('reference_ctc_deletion_margin'))}; "
        f"substitution_margin={_fmt(row.get('ctc_substitution_margin'))}; "
        f"logit_margin={_fmt(row.get('ctc_logit_margin'))}"
    )


def _fmt(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "nan"


def _first(group: pd.DataFrame, column: str) -> str:
    if column not in group.columns:
        return ""
    values = group[column].dropna().astype(str)
    return values.iloc[0] if len(values) else ""


def _sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, float(value)))
    return float(1.0 / (1.0 + math.exp(-value)))
