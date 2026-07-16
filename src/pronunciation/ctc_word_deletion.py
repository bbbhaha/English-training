from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import math
import threading

import numpy as np
import pandas as pd


DEFAULT_CTC_MODEL = "facebook/wav2vec2-base-960h"
_MODEL_LOCK = threading.Lock()


def ctc_sequence_log_probability(
    log_probs: object,
    labels: list[int] | np.ndarray,
    *,
    blank_id: int = 0,
) -> float:
    """Marginalize over all CTC alignments for one label sequence."""
    import torch
    import torch.nn.functional as functional

    values = torch.as_tensor(log_probs, dtype=torch.float32)
    if values.ndim != 2:
        raise ValueError("log_probs must have shape [time, vocabulary]")
    target = torch.as_tensor(labels, dtype=torch.long)
    if target.numel() == 0:
        return float(values[:, blank_id].sum().item())
    loss = functional.ctc_loss(
        values.unsqueeze(1),
        target,
        input_lengths=torch.tensor([values.shape[0]], dtype=torch.long),
        target_lengths=torch.tensor([target.numel()], dtype=torch.long),
        blank=blank_id,
        reduction="sum",
        zero_infinity=True,
    )
    return float(-loss.item())


def score_deletion_hypotheses(
    log_probs: object,
    full_labels: list[int],
    word_label_spans: list[tuple[int, int]],
    *,
    blank_id: int = 0,
    temperature: float = 3.0,
) -> list[dict[str, float]]:
    """Compare the canonical sequence with one-word-deleted alternatives."""
    full_log_probability = ctc_sequence_log_probability(log_probs, full_labels, blank_id=blank_id)
    rows: list[dict[str, float]] = []
    for start, end in word_label_spans:
        deleted_labels = full_labels[:start] + full_labels[end:]
        deleted_log_probability = ctc_sequence_log_probability(
            log_probs,
            deleted_labels,
            blank_id=blank_id,
        )
        removed_count = max(1, end - start)
        margin = (deleted_log_probability - full_log_probability) / removed_count
        score = _sigmoid(margin / max(float(temperature), 1e-6))
        rows.append(
            {
                "ctc_full_log_probability": full_log_probability,
                "ctc_deleted_log_probability": deleted_log_probability,
                "ctc_deletion_margin": margin,
                "ctc_deletion_score": score,
            }
        )
    return rows


def score_audio_word_deletions(
    audio_path: Path,
    target_text: str,
    *,
    model_id: str = DEFAULT_CTC_MODEL,
    local_files_only: bool = True,
) -> pd.DataFrame:
    """Return alignment-free character-CTC deletion evidence for every word."""
    from pronunciation.target_words import build_target_word_table

    words = build_target_word_table(target_text)
    columns = _output_columns()
    if words.empty:
        return pd.DataFrame(columns=columns)
    try:
        processor, model = _load_model(model_id, local_files_only)
        samples = _read_audio_16k(Path(audio_path))
        normalized_words = words["normalized_word"].fillna(words["word"]).astype(str).tolist()
        full_text = " ".join(normalized_words)
        full_labels = list(processor.tokenizer(full_text, add_special_tokens=False).input_ids)
        spans = _word_token_spans(processor.tokenizer, normalized_words)
        inputs = processor(samples, sampling_rate=16000, return_tensors="pt")
        import torch

        with _MODEL_LOCK, torch.inference_mode():
            logits = model(inputs.input_values).logits.squeeze(0)
            log_probs = logits.log_softmax(dim=-1).cpu()
        blank_id = int(model.config.pad_token_id or 0)
        scores = score_deletion_hypotheses(log_probs, full_labels, spans, blank_id=blank_id)
        greedy_ids = torch.argmax(log_probs, dim=-1)
        transcript = processor.batch_decode(greedy_ids.unsqueeze(0))[0]
        rows = []
        for (_, word), score in zip(words.iterrows(), scores):
            rows.append(
                {
                    "word_index": int(word["word_index"]),
                    "word": str(word["word"]),
                    **score,
                    "ctc_deletion_available": True,
                    "ctc_deletion_model": model_id,
                    "ctc_greedy_transcript": transcript,
                    "ctc_deletion_error": "",
                }
            )
        return pd.DataFrame(rows, columns=columns)
    except Exception as error:
        return pd.DataFrame(
            [
                {
                    "word_index": int(row["word_index"]),
                    "word": str(row["word"]),
                    "ctc_full_log_probability": float("nan"),
                    "ctc_deleted_log_probability": float("nan"),
                    "ctc_deletion_margin": float("nan"),
                    "ctc_deletion_score": float("nan"),
                    "ctc_deletion_available": False,
                    "ctc_deletion_model": model_id,
                    "ctc_greedy_transcript": "",
                    "ctc_deletion_error": f"{type(error).__name__}: {error}",
                }
                for _, row in words.iterrows()
            ],
            columns=columns,
        )


@lru_cache(maxsize=4)
def _load_model(model_id: str, local_files_only: bool):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(model_id, local_files_only=local_files_only)
    model = Wav2Vec2ForCTC.from_pretrained(model_id, local_files_only=local_files_only)
    model.eval()
    return processor, model


def _word_token_spans(tokenizer, words: list[str]) -> list[tuple[int, int]]:
    full_ids = list(tokenizer(" ".join(words), add_special_tokens=False).input_ids)
    spans: list[tuple[int, int]] = []
    cursor = 0
    delimiter_id = getattr(tokenizer, "word_delimiter_token_id", None)
    for index, word in enumerate(words):
        word_ids = list(tokenizer(word, add_special_tokens=False).input_ids)
        while cursor < len(full_ids) and delimiter_id is not None and full_ids[cursor] == delimiter_id:
            cursor += 1
        start = cursor
        cursor += len(word_ids)
        end = cursor
        if index < len(words) - 1 and delimiter_id is not None and cursor < len(full_ids):
            end += 1
            cursor += 1
        spans.append((start, end))
    return spans


def _read_audio_16k(path: Path) -> np.ndarray:
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    sample_rate, samples = wavfile.read(path)
    values = np.asarray(samples)
    if values.ndim == 2:
        values = values.mean(axis=1)
    if np.issubdtype(values.dtype, np.integer):
        scale = float(max(abs(np.iinfo(values.dtype).min), np.iinfo(values.dtype).max))
        values = values.astype(np.float32) / scale
    else:
        values = values.astype(np.float32)
    if int(sample_rate) != 16000:
        divisor = math.gcd(int(sample_rate), 16000)
        values = resample_poly(values, 16000 // divisor, int(sample_rate) // divisor).astype(np.float32)
    return values


def _sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, float(value)))
    return float(1.0 / (1.0 + math.exp(-value)))


def _output_columns() -> list[str]:
    return [
        "word_index",
        "word",
        "ctc_full_log_probability",
        "ctc_deleted_log_probability",
        "ctc_deletion_margin",
        "ctc_deletion_score",
        "ctc_deletion_available",
        "ctc_deletion_model",
        "ctc_greedy_transcript",
        "ctc_deletion_error",
    ]
