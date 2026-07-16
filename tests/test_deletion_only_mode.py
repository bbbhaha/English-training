from argparse import Namespace
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from predict_pronunciation import _final_output
from pronunciation.decision import DecisionConfig, apply_decision_rules
from pronunciation.deletion_detector import build_word_summary, detect_word_deletions


class DeletionOnlyModeTests(unittest.TestCase):
    def test_complete_reading_defaults_to_correct(self):
        frame = pd.DataFrame(
            {
                "alignment_quality": ["pass", "pass", "pass"],
                "prob_correct": [0.1, 0.2, 0.3],
                "manual_calibrated_error_probability": [1.0, 1.0, 1.0],
                "confidence": [1.0, 1.0, 1.0],
                "possible_missing_word": [False, False, False],
            }
        )
        out = apply_decision_rules(frame, DecisionConfig(mode="deletion_only"))
        self.assertTrue(out["decision"].eq("correct").all())
        self.assertTrue(out["error_type"].eq("").all())

    def test_high_error_probability_does_not_create_true_error(self):
        frame = pd.DataFrame(
            {
                "alignment_quality": ["pass"],
                "model_error_score": [1.0],
                "manual_calibrated_error_probability": [1.0],
                "confidence": [1.0],
                "prob_correct": [0.0],
                "possible_missing_word": [False],
            }
        )
        out = apply_decision_rules(frame, DecisionConfig(mode="deletion_only"))
        self.assertEqual(out.loc[0, "decision"], "correct")
        self.assertNotEqual(out.loc[0, "decision"], "true_error")

    def test_possible_missing_word_is_review_possible_deletion(self):
        frame = pd.DataFrame(
            {
                "alignment_quality": ["pass", "pass"],
                "possible_missing_word": [True, True],
                "missing_word_reason": ["high_error_ratio_ge_0.8", "high_error_ratio_ge_0.8"],
                "deletion_trigger_source": ["extreme_duration_compression", "extreme_duration_compression"],
            }
        )
        out = apply_decision_rules(frame, DecisionConfig(mode="deletion_only"))
        self.assertTrue(out["alignment_quality"].eq("suspect").all())
        self.assertTrue(out["decision"].eq("uncertain_review").all())
        self.assertTrue(out["error_type"].eq("possible_deletion").all())
        self.assertTrue(out["review_reason"].str.contains("possible_missing_word").all())

    def test_detect_deletion_as_error_outputs_deletion(self):
        frame = pd.DataFrame(
            {
                "alignment_quality": ["pass"],
                "possible_missing_word": [True],
                "missing_word_reason": ["word_duration_lt_250ms"],
                "deletion_trigger_source": ["extreme_duration_compression"],
            }
        )
        out = apply_decision_rules(
            frame,
            DecisionConfig(mode="deletion_only", detect_deletion_as_error=True),
        )
        self.assertEqual(out.loc[0, "decision"], "true_error")
        self.assertEqual(out.loc[0, "error_type"], "deletion")
        self.assertEqual(out.loc[0, "review_reason"], "missing_word_detected")

    def test_deletion_only_ignores_hardset_thresholds(self):
        frame = pd.DataFrame(
            {
                "alignment_quality": ["pass"],
                "model_error_score": [0.99],
                "manual_calibrated_error_probability": [0.99],
                "confidence": [0.99],
                "possible_missing_word": [False],
            }
        )
        out = apply_decision_rules(
            frame,
            DecisionConfig(
                mode="deletion_only",
                hardset_model_error_threshold=0.01,
                hardset_probability_threshold=0.01,
            ),
        )
        self.assertEqual(out.loc[0, "decision"], "correct")

    def test_word_summary_deletion_only_outputs_expected_types(self):
        frame = pd.DataFrame(
            {
                "word": ["AMERICA", "AMERICA", "GREAT"],
                "word_index": [1, 1, 2],
                "start_ms": [0, 40, 200],
                "end_ms": [40, 80, 300],
                "duration_ms": [40, 40, 100],
                "alignment_quality": ["suspect", "suspect", "pass"],
                "possible_missing_word": [True, True, False],
                "missing_word_reason": ["high_error_ratio_ge_0.8", "high_error_ratio_ge_0.8", ""],
                "decision": ["uncertain_review", "uncertain_review", "correct"],
                "error_type": ["possible_deletion", "possible_deletion", ""],
            }
        )
        summary = build_word_summary(frame)
        america = summary[summary["word"].eq("AMERICA")].iloc[0]
        great = summary[summary["word"].eq("GREAT")].iloc[0]
        self.assertEqual(america["word_decision"], "uncertain_review")
        self.assertEqual(america["error_type"], "possible_deletion")
        self.assertEqual(great["word_decision"], "correct")
        self.assertEqual(great["error_type"], "")

    def test_predict_final_output_accepts_deletion_only(self):
        frame = pd.DataFrame({"alignment_quality": ["pass"], "possible_missing_word": [False]})
        args = Namespace(
            decision_mode="deletion_only",
            main_error_threshold=0.01,
            true_error_threshold=0.01,
            detect_deletion_as_error=False,
        )
        out = _final_output(frame, args)
        self.assertEqual(out.loc[0, "decision"], "correct")

    def test_webapp_keeps_deletion_only_optional_but_defaults_to_phone_diagnosis(self):
        text = (ROOT / "webapp" / "app.py").read_text(encoding="utf-8")
        self.assertIn('decision_mode="phone_diagnosis"', text)
        self.assertNotIn('decision_mode="hardset"', text)

    def test_high_error_ratio_is_debug_only_not_missing_word(self):
        frame = _word_frame_with_duration(400, error_prob=1.0)
        out, _ = detect_word_deletions(frame, mode="deletion_only")
        self.assertFalse(bool(out["possible_missing_word"].any()))
        self.assertTrue(out["alignment_quality"].eq("pass").all())
        summary = build_word_summary(out, mode="deletion_only")
        self.assertEqual(summary.loc[0, "word_decision"], "correct")
        self.assertEqual(summary.loc[0, "deletion_trigger_source"], "none")
        self.assertIn("debug_high_error_ratio_ge_0.8", summary.loc[0, "debug_reason"])

    def test_america_400ms_high_error_is_not_missing_word(self):
        frame = _word_frame_with_duration(400, phone_count=7, error_prob=1.0)
        out, summary = detect_word_deletions(frame, mode="deletion_only")
        self.assertFalse(bool(out["possible_missing_word"].any()))
        self.assertFalse(bool(summary.loc[0, "possible_missing_word"]))
        self.assertEqual(summary.loc[0, "deletion_trigger_source"], "none")

    def test_america_80ms_triggers_extreme_duration_compression(self):
        frame = _word_frame_with_duration(80, phone_count=7, error_prob=0.1)
        out, summary = detect_word_deletions(frame, mode="deletion_only")
        self.assertTrue(bool(out["possible_missing_word"].all()))
        self.assertEqual(summary.loc[0, "deletion_trigger_source"], "extreme_duration_compression")
        self.assertIn("extreme_word_duration_compression", summary.loc[0, "missing_word_reason"])

    def test_asr_missing_word_triggers_missing_word(self):
        frame = _word_frame_with_duration(400, phone_count=7, error_prob=0.1)
        frame["asr_missing_word"] = True
        out, summary = detect_word_deletions(frame, mode="deletion_only")
        self.assertTrue(bool(out["possible_missing_word"].all()))
        self.assertEqual(summary.loc[0, "deletion_trigger_source"], "asr_missing_word")

    def test_missing_boundary_ratio_one_triggers_missing_boundaries(self):
        frame = _word_frame_with_duration(400, phone_count=7, error_prob=0.1)
        frame["start_ms"] = pd.NA
        frame["end_ms"] = pd.NA
        out, summary = detect_word_deletions(frame, mode="deletion_only")
        self.assertTrue(bool(out["possible_missing_word"].all()))
        self.assertEqual(summary.loc[0, "deletion_trigger_source"], "missing_boundaries")


if __name__ == "__main__":
    unittest.main()


def _word_frame_with_duration(total_ms: float, phone_count: int = 7, error_prob: float = 1.0) -> pd.DataFrame:
    step = total_ms / phone_count
    starts = [round(i * step, 3) for i in range(phone_count)]
    ends = [round((i + 1) * step, 3) for i in range(phone_count)]
    return pd.DataFrame(
        {
            "word": ["AMERICA"] * phone_count,
            "word_index": [1] * phone_count,
            "target_phone": ["AH", "M", "EH", "R", "IH", "K", "AH"][:phone_count],
            "start_ms": starts,
            "end_ms": ends,
            "duration_ms": [step] * phone_count,
            "alignment_quality": ["pass"] * phone_count,
            "manual_calibrated_error_probability": [error_prob] * phone_count,
            "prob_correct": [0.1 if error_prob > 0.8 else 0.9] * phone_count,
            "decision": ["correct"] * phone_count,
            "error_type": [""] * phone_count,
        }
    )
