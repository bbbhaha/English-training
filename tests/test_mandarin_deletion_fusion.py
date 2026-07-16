from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from scipy.io import wavfile
from sklearn.dummy import DummyClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from phoneme_assessment.textgrid import Interval
from pronunciation.mandarin_deletion_fusion import (
    FEATURE_COLUMNS,
    add_mandarin_deletion_fusion_scores,
    build_mandarin_deletion_features,
)
from pronunciation.word_deletion_model import word_deletion_detector
from scripts.train_mandarin_deletion_fusion import synthesize_word_deletion


class MandarinDeletionFusionTests(unittest.TestCase):
    def test_feature_builder_tracks_two_recognizer_agreement(self) -> None:
        features = build_mandarin_deletion_features(_evidence_frame())
        self.assertEqual(FEATURE_COLUMNS, features.columns.tolist())
        self.assertEqual(1.0, features.loc[0, "recognizer_deletion_agreement"])
        self.assertEqual(0.0, features.loc[1, "recognizer_deletion_agreement"])
        self.assertEqual(1.0, features.loc[0, "is_function_word"])

    def test_artifact_probability_is_added(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "fusion.joblib"
            _write_constant_artifact(model_path, constant=1)
            scored = add_mandarin_deletion_fusion_scores(_evidence_frame(), model_path)
        self.assertTrue(scored["mandarin_deletion_model_available"].all())
        self.assertTrue(scored["mandarin_deletion_probability"].ge(0.99).all())

    def test_fusion_model_can_confirm_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "fusion.joblib"
            _write_constant_artifact(model_path, constant=1)
            diagnosed = word_deletion_detector(
                _evidence_frame().iloc[[0]],
                mandarin_fusion_model=model_path,
            )
        self.assertEqual("deletion", diagnosed.loc[diagnosed.index[0], "deletion_decision"])
        self.assertIn("mandarin_l1_fusion", diagnosed.loc[diagnosed.index[0], "deletion_evidence"])

    def test_low_fusion_probability_rejects_asr_only_false_alarm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "fusion.joblib"
            _write_constant_artifact(model_path, constant=0)
            diagnosed = word_deletion_detector(
                _evidence_frame().iloc[[0]],
                mandarin_fusion_model=model_path,
            )
        self.assertEqual("correct", diagnosed.loc[diagnosed.index[0], "deletion_decision"])

    def test_synthetic_deletion_removes_interval_with_crossfade(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.wav"
            output = Path(directory) / "deleted.wav"
            samples = (np.sin(np.linspace(0, 40, 16000)) * 12000).astype(np.int16)
            wavfile.write(source, 16000, samples)
            synthesize_word_deletion(source, output, Interval(0.30, 0.50, "WORD"))
            sample_rate, deleted = wavfile.read(output)
        self.assertEqual(16000, sample_rate)
        self.assertLess(len(deleted), len(samples) - 2500)

    def test_ambiguous_single_phone_function_word_is_only_possible(self) -> None:
        frame = _evidence_frame().iloc[[0]].copy()
        frame["word"] = "A"
        frame["phone_count"] = 1
        frame["ctc_deletion_score"] = 0.87
        frame["ctc_deletion_margin"] = 5.7
        frame["ctc_greedy_missing_word"] = False
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "fusion.joblib"
            _write_constant_artifact(model_path, constant=1)
            diagnosed = word_deletion_detector(frame, mandarin_fusion_model=model_path)
        self.assertEqual("possible_deletion", diagnosed.iloc[0]["deletion_decision"])
        self.assertIn("short_function_word", diagnosed.iloc[0]["deletion_evidence"])


def _evidence_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "utterance_id": "u1", "word_index": 1, "word": "THE", "phone_count": 2,
                "word_duration_ms": float("nan"), "alignment_quality": "pass", "lexicon_status": "cmudict",
                "asr_missing_word": True, "asr_substituted_word": False, "asr_confidence": 0.9,
                "asr_context_support": 1.0, "asr_missing_confidence": 0.9,
                "ctc_deletion_available": True, "ctc_deletion_score": 0.95, "ctc_deletion_margin": 8.0,
                "ctc_greedy_missing_word": True, "ctc_greedy_substituted_word": False,
                "ctc_greedy_context_support": 1.0,
            },
            {
                "utterance_id": "u1", "word_index": 2, "word": "BIRD", "phone_count": 3,
                "word_duration_ms": 320.0, "alignment_quality": "pass", "lexicon_status": "cmudict",
                "asr_missing_word": False, "asr_substituted_word": False, "asr_confidence": 0.9,
                "asr_context_support": 1.0, "asr_missing_confidence": 0.0,
                "ctc_deletion_available": True, "ctc_deletion_score": 0.05, "ctc_deletion_margin": -8.0,
                "ctc_greedy_missing_word": False, "ctc_greedy_substituted_word": False,
                "ctc_greedy_context_support": 1.0,
            },
        ]
    )


def _write_constant_artifact(path: Path, *, constant: int) -> None:
    model = DummyClassifier(strategy="constant", constant=constant)
    model.fit(pd.DataFrame(np.zeros((2, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS), [0, 1])
    joblib.dump(
        {
            "name": "test_fusion",
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "thresholds": {"deletion": 0.80, "possible_deletion": 0.45},
        },
        path,
    )


if __name__ == "__main__":
    unittest.main()
