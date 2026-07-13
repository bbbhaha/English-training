from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pronunciation.decision import DecisionConfig, apply_decision_rules
from pronunciation.deletion_detector import build_word_summary, detect_word_deletions


def _word_frame(
    durations: list[float],
    *,
    word: str = "AMERICA",
    starts: list[float] | None = None,
    error_probs: list[float] | None = None,
    prob_correct: list[float] | None = None,
) -> pd.DataFrame:
    starts = starts if starts is not None else [i * 20.0 for i in range(len(durations))]
    error_probs = error_probs if error_probs is not None else [0.2] * len(durations)
    prob_correct = prob_correct if prob_correct is not None else [0.8] * len(durations)
    return pd.DataFrame(
        {
            "word": [word] * len(durations),
            "word_index": [1] * len(durations),
            "target_phone": ["AH", "M", "EH", "R", "IH", "K", "AH"][: len(durations)],
            "phone_index": list(range(len(durations))),
            "start_ms": starts,
            "end_ms": [start + dur if pd.notna(start) else pd.NA for start, dur in zip(starts, durations)],
            "duration_ms": durations,
            "alignment_quality": ["pass"] * len(durations),
            "manual_calibrated_error_probability": error_probs,
            "prob_correct": prob_correct,
            "confidence": [0.9] * len(durations),
            "calibration_available": [True] * len(durations),
        }
    )


class DeletionDetectorTests(unittest.TestCase):
    def test_multi_phone_word_short_total_duration_is_possible_missing_word(self):
        frame = _word_frame([20, 20, 20, 20], starts=[0, 20, 40, 60])
        out, summary = detect_word_deletions(frame)
        self.assertTrue(bool(out["possible_missing_word"].all()))
        self.assertIn("word_duration_lt_250ms", out.loc[0, "missing_word_reason"])
        self.assertEqual(out.loc[0, "alignment_quality"], "suspect")
        self.assertNotEqual(out.loc[0, "alignment_quality"], "pass")
        self.assertTrue(bool(summary.loc[0, "possible_missing_word"]))

    def test_many_short_phone_durations_are_possible_missing_word(self):
        frame = _word_frame([20, 25, 20, 90, 100], starts=[0, 40, 80, 200, 330])
        out, _ = detect_word_deletions(frame)
        self.assertTrue(bool(out["possible_missing_word"].all()))
        self.assertIn("short_phone_ratio_ge_0.5", out.loc[0, "missing_word_reason"])

    def test_many_high_error_probs_are_possible_missing_word(self):
        frame = _word_frame([80, 80, 80, 80, 80], starts=[0, 90, 180, 270, 360], error_probs=[0.9] * 5)
        out, _ = detect_word_deletions(frame)
        self.assertTrue(bool(out["possible_missing_word"].all()))
        self.assertIn("high_error_ratio_ge_0.8", out.loc[0, "missing_word_reason"])

    def test_missing_boundaries_are_possible_missing_word(self):
        frame = _word_frame([80, 80, 80, 80], starts=[pd.NA, pd.NA, 200, 300])
        out, _ = detect_word_deletions(frame)
        self.assertTrue(bool(out["possible_missing_word"].all()))
        self.assertIn("missing_boundary_ratio_ge_0.5", out.loc[0, "missing_word_reason"])

    def test_possible_missing_word_forces_uncertain_or_deletion_decision(self):
        frame = _word_frame([20, 20, 20, 20], starts=[0, 20, 40, 60])
        detected, _ = detect_word_deletions(frame)
        out = apply_decision_rules(detected, DecisionConfig(mode="conservative"))
        self.assertTrue(out["decision"].eq("uncertain_review").all())
        self.assertTrue(out["error_type"].eq("possible_deletion").all())
        self.assertTrue(out["alignment_quality"].eq("suspect").all())

        hard_deletion = apply_decision_rules(
            detected,
            DecisionConfig(mode="conservative", detect_deletion_as_error=True),
        )
        self.assertTrue(hard_deletion["decision"].eq("true_error").all())
        self.assertTrue(hard_deletion["error_type"].eq("deletion").all())

    def test_word_summary_marks_possible_deletion(self):
        frame = _word_frame([20, 20, 20, 20], starts=[0, 20, 40, 60])
        detected, _ = detect_word_deletions(frame)
        out = apply_decision_rules(detected, DecisionConfig(mode="conservative"))
        summary = build_word_summary(out)
        self.assertTrue(bool(summary.loc[0, "possible_missing_word"]))
        self.assertEqual(summary.loc[0, "alignment_quality"], "suspect")
        self.assertEqual(summary.loc[0, "error_type"], "possible_deletion")

    def test_normal_multi_phone_word_is_not_flagged(self):
        frame = _word_frame([80, 90, 75, 85, 95], starts=[0, 90, 190, 275, 370], error_probs=[0.1] * 5)
        out, summary = detect_word_deletions(frame)
        self.assertFalse(bool(out["possible_missing_word"].any()))
        self.assertFalse(bool(summary.loc[0, "possible_missing_word"]))
        self.assertTrue(out["alignment_quality"].eq("pass").all())


if __name__ == "__main__":
    unittest.main()
