from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pronunciation.decision import DecisionConfig, apply_decision_rules, apply_deletion_only_override
from pronunciation.deletion_detector import build_word_summary, detect_word_deletions


def _phones(total_ms: float = 400.0, *, phone_count: int = 7, error_prob: float = 1.0) -> pd.DataFrame:
    step = total_ms / phone_count
    return pd.DataFrame(
        {
            "word": ["AMERICA"] * phone_count,
            "word_index": [1] * phone_count,
            "target_phone": ["AH", "M", "EH", "R", "IH", "K", "AH"][:phone_count],
            "start_ms": [i * step for i in range(phone_count)],
            "end_ms": [(i + 1) * step for i in range(phone_count)],
            "duration_ms": [step] * phone_count,
            "alignment_quality": ["pass"] * phone_count,
            "manual_calibrated_error_probability": [error_prob] * phone_count,
            "model_error_score": [1.0] * phone_count,
            "prob_correct": [0.0] * phone_count,
            "confidence": [0.0] * phone_count,
            "deletion_trigger_source": ["none"] * phone_count,
        }
    )


class DeletionOnlyStrictTests(unittest.TestCase):
    def test_error_scores_are_overridden_when_no_deletion_trigger(self):
        frame = _phones(error_prob=1.0)
        detected, _ = detect_word_deletions(frame, mode="deletion_only")
        out = apply_decision_rules(detected, DecisionConfig(mode="deletion_only"))
        summary = build_word_summary(out, mode="deletion_only")
        out, summary = apply_deletion_only_override(out, summary)

        self.assertTrue(out["decision"].eq("correct").all())
        self.assertTrue(out["manual_calibrated_error_probability"].eq(0.0).all())
        self.assertTrue(out["model_error_score"].eq(0.0).all())
        self.assertTrue(out["confidence"].eq(1.0).all())
        self.assertFalse(out["alignment_quality"].eq("suspect").any())
        self.assertFalse(bool(summary.loc[0, "possible_missing_word"]))
        self.assertEqual(summary.loc[0, "word_decision"], "correct")

    def test_normal_duration_high_error_probability_stays_correct(self):
        frame = _phones(total_ms=400, error_prob=1.0)
        detected, summary = detect_word_deletions(frame, mode="deletion_only")
        self.assertFalse(bool(detected["possible_missing_word"].any()))
        self.assertEqual(summary.loc[0, "deletion_trigger_source"], "none")
        out = apply_decision_rules(detected, DecisionConfig(mode="deletion_only"))
        self.assertTrue(out["decision"].eq("correct").all())
        self.assertTrue(out["alignment_quality"].isin(["pass", "good", "ok"]).all())

    def test_asr_missing_word_can_be_review_or_error(self):
        frame = _phones(total_ms=400, error_prob=1.0)
        frame["asr_missing_word"] = True
        detected, summary = detect_word_deletions(frame, mode="deletion_only")
        self.assertTrue(bool(summary.loc[0, "possible_missing_word"]))
        self.assertEqual(summary.loc[0, "deletion_trigger_source"], "asr_missing_word")

        review = apply_decision_rules(detected, DecisionConfig(mode="deletion_only"))
        self.assertTrue(review["decision"].eq("uncertain_review").all())
        self.assertTrue(review["error_type"].eq("possible_deletion").all())

        error = apply_decision_rules(detected, DecisionConfig(mode="deletion_only", detect_deletion_as_error=True))
        self.assertTrue(error["decision"].eq("true_error").all())
        self.assertTrue(error["error_type"].eq("deletion").all())

    def test_extreme_short_duration_triggers_deletion_source(self):
        frame = _phones(total_ms=80, error_prob=0.0)
        detected, summary = detect_word_deletions(frame, mode="deletion_only")
        self.assertTrue(bool(detected["possible_missing_word"].all()))
        self.assertEqual(summary.loc[0, "deletion_trigger_source"], "extreme_duration_compression")

    def test_deletion_only_error_display_is_not_probability(self):
        app_js = (ROOT / "webapp" / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("function errorDisplay(row)", app_js)
        self.assertIn('return row.display_error || "0%"', app_js)
        self.assertNotIn("row.manual_calibrated_error_probability", app_js)


if __name__ == "__main__":
    unittest.main()
