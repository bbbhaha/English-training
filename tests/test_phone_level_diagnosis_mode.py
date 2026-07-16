from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "webapp"))

from pronunciation.g2p import text_to_phones
from pronunciation.phone_decision import apply_phone_decisions, compute_phone_decision, summarize_phone_decisions
from app import apply_phone_diagnosis_display


class PhoneLevelDiagnosisModeTests(unittest.TestCase):
    def test_low_error_probability_is_correct(self):
        result = compute_phone_decision({
            "phone_error_probability": 0.10,
            "manual_calibrated_error_probability": 0.10,
            "alignment_quality": "pass",
        })
        self.assertEqual(result["phone_decision"], "correct")
        self.assertEqual(result["phone_error_type"], "")

    def test_high_error_probability_is_true_error(self):
        result = compute_phone_decision({
            "manual_calibrated_error_probability": 0.82,
            "alignment_quality": "pass",
        })
        self.assertEqual(result["phone_decision"], "true_error")
        self.assertEqual(result["phone_error_type"], "mispronunciation")

    def test_medium_error_probability_needs_review(self):
        result = compute_phone_decision({
            "manual_calibrated_error_probability": 0.60,
            "alignment_quality": "pass",
        })
        self.assertEqual(result["phone_decision"], "uncertain_review")
        self.assertEqual(result["phone_error_type"], "possible_mispronunciation")

    def test_bad_alignment_has_priority(self):
        result = compute_phone_decision({
            "manual_calibrated_error_probability": 0.10,
            "alignment_quality": "bad",
        })
        self.assertEqual(result["phone_decision"], "uncertain_review")
        self.assertEqual(result["phone_error_type"], "alignment_issue")
        self.assertGreaterEqual(result["phone_error_probability"], 0.50)

    def test_saturated_manual_calibration_cannot_override_high_prob_correct(self):
        result = compute_phone_decision({
            "manual_calibrated_error_probability": 1.0,
            "model_error_score": 0.06,
            "prob_correct": 0.94,
            "alignment_quality": "pass",
        })
        self.assertEqual(result["phone_decision"], "correct")
        self.assertEqual(result["phone_error_probability"], 0.06)
        self.assertEqual(result["phone_score_source"], "model_error_score")

    def test_reapplying_phone_decisions_does_not_reuse_saturated_probability(self):
        frame = pd.DataFrame([{
            "manual_calibrated_error_probability": 1.0,
            "model_error_score": 0.08,
            "prob_correct": 0.92,
            "alignment_quality": "pass",
        }])
        result = apply_phone_decisions(apply_phone_decisions(frame))
        self.assertEqual(result.loc[0, "phone_decision"], "correct")
        self.assertEqual(result.loc[0, "phone_error_probability"], 0.08)
        self.assertEqual(result.loc[0, "phone_score_source"], "model_error_score")

    def test_sentence_g2p_contains_every_target_phone(self):
        result = text_to_phones("SHE SEES THE BLUE BIRD")
        words = {row["word_index"] for row in result.phones}
        self.assertEqual(words, {0, 1, 2, 3, 4})
        self.assertGreater(len(result.phones), 5)

    def test_web_display_keeps_one_row_per_phone(self):
        prediction = pd.DataFrame([
            {"word": "SHE", "target_phone": "SH", "word_index": 0, "phone_index": 0,
             "manual_calibrated_error_probability": 0.10, "alignment_quality": "pass"},
            {"word": "SHE", "target_phone": "IY", "word_index": 0, "phone_index": 1,
             "manual_calibrated_error_probability": 0.82, "alignment_quality": "pass"},
        ])
        diagnosed = apply_phone_decisions(prediction)
        display = apply_phone_diagnosis_display(diagnosed)
        self.assertEqual(len(display), 2)
        self.assertEqual(display["target_phone"].tolist(), ["SH", "IY"])

    def test_word_summary_only_aggregates_phone_results(self):
        phones = apply_phone_decisions(pd.DataFrame([
            {"word": "SHE", "word_index": 0, "target_phone": "SH",
             "manual_calibrated_error_probability": 0.10, "alignment_quality": "pass"},
            {"word": "SHE", "word_index": 0, "target_phone": "IY",
             "manual_calibrated_error_probability": 0.82, "alignment_quality": "pass"},
        ]))
        summary = summarize_phone_decisions(phones)
        self.assertEqual(summary.iloc[0]["word_decision"], "has_phone_error")
        self.assertEqual(summary.iloc[0]["phone_count"], 2)


if __name__ == "__main__":
    unittest.main()
