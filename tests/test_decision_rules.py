from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from debug_prediction_scores import analyze_prediction_scores
from pronunciation.calibration import apply_manual_calibrator
from pronunciation.decision import DecisionConfig, apply_decision_rules, is_good_alignment


class DecisionRuleTests(unittest.TestCase):
    def test_pass_alignment_is_good(self):
        self.assertTrue(is_good_alignment("pass"))
        self.assertTrue(is_good_alignment("good"))
        self.assertTrue(is_good_alignment("ok"))
        self.assertFalse(is_good_alignment("bad"))

    def test_high_prob_correct_cannot_be_true_error_in_conservative_mode(self):
        frame = pd.DataFrame(
            {
                "alignment_quality": ["pass"],
                "prob_correct": [0.96],
                "model_error_score": [0.04],
                "manual_calibrated_error_probability": [1.0],
                "confidence": [0.96],
                "calibration_available": [True],
            }
        )
        out = apply_decision_rules(frame, DecisionConfig(mode="conservative"))
        self.assertEqual(out.loc[0, "decision"], "correct")

    def test_missing_calibration_defaults_to_half_and_no_true_error(self):
        frame = pd.DataFrame({"model_error_score": [0.9], "prob_correct": [0.1]})
        out = apply_manual_calibrator(frame, Path("missing_calibrator_for_test.joblib"))
        self.assertEqual(float(out.loc[0, "manual_calibrated_error_probability"]), 0.5)
        self.assertFalse(bool(out.loc[0, "calibration_available"]))
        decided = apply_decision_rules(
            out.assign(alignment_quality="pass", confidence=0.9),
            DecisionConfig(mode="conservative"),
        )
        self.assertNotEqual(decided.loc[0, "decision"], "true_error")

    def test_debug_flags_probability_saturation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prediction.csv"
            pd.DataFrame(
                {
                    "decision": ["true_error", "true_error"],
                    "alignment_quality": ["pass", "pass"],
                    "model_error_score": [0.2, 0.2],
                    "prob_correct": [0.8, 0.9],
                    "manual_calibrated_error_probability": [1.0, 1.0],
                    "confidence": [0.8, 0.9],
                }
            ).to_csv(path, index=False)
            report = analyze_prediction_scores(path)
        self.assertTrue(report["possible_probability_saturation_bug"])
        self.assertTrue(report["possible_fallback_model_bug"])
        self.assertEqual(report["high_prob_correct_but_true_error_count"], 2)

    def test_conservative_correct_sample_not_mass_true_error(self):
        frame = pd.DataFrame(
            {
                "alignment_quality": ["pass", "pass", "pass"],
                "prob_correct": [0.90, 0.82, 0.70],
                "model_error_score": [0.10, 0.18, 0.30],
                "manual_calibrated_error_probability": [1.0, 0.99, 0.70],
                "confidence": [0.90, 0.82, 0.70],
                "calibration_available": [True, True, True],
            }
        )
        out = apply_decision_rules(frame, DecisionConfig(mode="conservative"))
        self.assertEqual(out["decision"].tolist()[:2], ["correct", "correct"])
        self.assertNotEqual(out.loc[2, "decision"], "true_error")


if __name__ == "__main__":
    unittest.main()

