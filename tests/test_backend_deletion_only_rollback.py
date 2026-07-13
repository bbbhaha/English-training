from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "webapp"))

from app import deletion_only_display_fields
from pronunciation.decision import apply_deletion_only_override
from pronunciation.deletion_detector import build_word_summary, detect_word_deletions


def _prediction(total_ms: float = 400.0, *, phone_count: int = 7, source: str = "none") -> pd.DataFrame:
    step = total_ms / phone_count
    return pd.DataFrame(
        {
            "word": ["AMERICA"] * phone_count,
            "word_index": [1] * phone_count,
            "target_phone": ["AH", "M", "EH", "R", "IH", "K", "AH"][:phone_count],
            "phone_index": list(range(phone_count)),
            "start_ms": [i * step for i in range(phone_count)],
            "end_ms": [(i + 1) * step for i in range(phone_count)],
            "duration_ms": [step] * phone_count,
            "alignment_quality": ["pass"] * phone_count,
            "manual_calibrated_error_probability": [1.0] * phone_count,
            "model_error_score": [1.0] * phone_count,
            "prob_correct": [0.0] * phone_count,
            "confidence": [0.0] * phone_count,
            "high_error_ratio": [1.0] * phone_count,
            "deletion_trigger_source": [source] * phone_count,
            "possible_missing_word": [source != "none"] * phone_count,
            "missing_word_reason": [source if source != "none" else ""] * phone_count,
        }
    )


class BackendDeletionOnlyRollbackTests(unittest.TestCase):
    def test_scores_do_not_affect_default_backend(self):
        pred = _prediction(source="none")
        summary = build_word_summary(pred, mode="deletion_only")
        out, summary = apply_deletion_only_override(pred, summary)

        self.assertTrue(out["decision"].eq("correct").all())
        self.assertTrue(out["manual_calibrated_error_probability"].eq(0.0).all())
        self.assertTrue(out["model_error_score"].eq(0.0).all())
        self.assertTrue(out["confidence"].eq(1.0).all())
        self.assertFalse(bool(out["possible_missing_word"].any()))
        self.assertFalse(out["alignment_quality"].eq("suspect").any())
        self.assertEqual(summary.loc[0, "word_decision"], "correct")
        self.assertFalse(bool(summary.loc[0, "possible_missing_word"]))

    def test_complete_reading_frontend_error_is_zero(self):
        pred = _prediction(source="none")
        summary = build_word_summary(pred, mode="deletion_only")
        out, summary = apply_deletion_only_override(pred, summary)
        display = deletion_only_display_fields(out.iloc[0].to_dict())

        self.assertTrue(out["decision"].eq("correct").all())
        self.assertEqual(display["error_display"], "0%")
        self.assertEqual(summary.loc[0, "word_decision"], "correct")

    def test_asr_missing_review_mode(self):
        pred = _prediction(source="asr_missing_word")
        summary = build_word_summary(pred, mode="deletion_only")
        out, summary = apply_deletion_only_override(pred, summary, detect_deletion_as_error=False)

        self.assertTrue(out["decision"].eq("uncertain_review").all())
        self.assertTrue(out["error_type"].eq("possible_deletion").all())
        self.assertTrue(out["alignment_quality"].eq("suspect").all())
        self.assertEqual(summary.loc[0, "word_decision"], "uncertain_review")
        self.assertEqual(summary.loc[0, "error_type"], "possible_deletion")

    def test_asr_missing_error_mode(self):
        pred = _prediction(source="asr_missing_word")
        summary = build_word_summary(pred, mode="deletion_only")
        out, summary = apply_deletion_only_override(pred, summary, detect_deletion_as_error=True)

        self.assertTrue(out["decision"].eq("true_error").all())
        self.assertTrue(out["error_type"].eq("deletion").all())
        self.assertEqual(summary.loc[0, "word_decision"], "true_error")
        self.assertEqual(summary.loc[0, "error_type"], "deletion")

    def test_extreme_duration_compression_detector(self):
        pred = _prediction(total_ms=60, source="none")
        detected, summary = detect_word_deletions(pred, mode="deletion_only")

        self.assertTrue(bool(detected["possible_missing_word"].all()))
        self.assertEqual(summary.loc[0, "deletion_trigger_source"], "extreme_duration_compression")

    def test_normal_duration_high_error_prob_not_missing(self):
        pred = _prediction(total_ms=400, source="none")
        detected, summary = detect_word_deletions(pred, mode="deletion_only")
        out, summary = apply_deletion_only_override(detected, build_word_summary(detected, mode="deletion_only"))

        self.assertFalse(bool(detected["possible_missing_word"].any()))
        self.assertEqual(summary.loc[0, "word_decision"], "correct")
        self.assertEqual(summary.loc[0, "deletion_trigger_source"], "none")
        self.assertTrue(out["decision"].eq("correct").all())

    def test_word_summary_has_required_debug_columns(self):
        pred = _prediction(total_ms=400, source="none")
        summary = build_word_summary(pred, mode="deletion_only")
        for col in [
            "word",
            "word_index",
            "phone_count",
            "word_duration_ms",
            "possible_missing_word",
            "missing_word_reason",
            "deletion_trigger_source",
            "alignment_quality",
            "word_decision",
            "error_type",
            "num_phone_true_error",
            "num_phone_uncertain",
            "debug_high_error_ratio",
            "debug_low_prob_correct_ratio",
            "debug_short_phone_ratio",
        ]:
            self.assertIn(col, summary.columns)


if __name__ == "__main__":
    unittest.main()
