from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_correct_audio_sanity import evaluate_manifest
from pronunciation.decision import DecisionConfig, apply_decision_rules


class ConservativeDecisionTests(unittest.TestCase):
    def test_high_prob_correct_cannot_be_true_error(self):
        frame = pd.DataFrame(
            {
                "alignment_quality": ["pass"],
                "prob_correct": [0.92],
                "manual_calibrated_error_probability": [0.99],
                "confidence": [0.95],
                "calibration_available": [True],
                "phonological_relation": ["likely_true_error"],
            }
        )
        out = apply_decision_rules(frame, DecisionConfig(mode="conservative"))
        self.assertEqual(out.loc[0, "decision"], "correct")

    def test_acceptable_variant_cannot_be_true_error(self):
        frame = pd.DataFrame(
            {
                "alignment_quality": ["pass"],
                "prob_correct": [0.2],
                "manual_calibrated_error_probability": [0.99],
                "confidence": [0.95],
                "calibration_available": [True],
                "phonological_relation": ["acceptable_variant"],
            }
        )
        out = apply_decision_rules(frame, DecisionConfig(mode="conservative"))
        self.assertEqual(out.loc[0, "decision"], "acceptable_accent")
        self.assertNotEqual(out.loc[0, "decision"], "true_error")

    def test_conservative_outputs_fewer_true_errors_than_hardset(self):
        frame = pd.DataFrame(
            {
                "alignment_quality": ["pass", "pass", "pass"],
                "prob_correct": [0.8, 0.5, 0.2],
                "model_error_score": [0.2, 0.5, 0.8],
                "manual_calibrated_error_probability": [0.95, 0.95, 0.95],
                "confidence": [0.8, 0.8, 0.8],
                "calibration_available": [True, True, True],
                "phonological_relation": ["", "", "likely_true_error"],
            }
        )
        conservative = apply_decision_rules(frame, DecisionConfig(mode="conservative"))
        hardset = apply_decision_rules(frame, DecisionConfig(mode="hardset"))
        self.assertLess(conservative["decision"].eq("true_error").sum(), hardset["decision"].eq("true_error").sum())

    def test_webapp_default_uses_deletion_only_for_demo(self):
        text = (ROOT / "webapp" / "app.py").read_text(encoding="utf-8")
        self.assertIn('decision_mode="deletion_only"', text)
        self.assertNotIn('decision_mode="hardset"', text)

    def test_correct_audio_sanity_outputs_false_positive_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "correct_audio_manifest.csv"
            output_dir = root / "out"
            output_dir.mkdir()
            pd.DataFrame(
                {
                    "utt_id": ["u1"],
                    "audio": ["dummy.wav"],
                    "text": ["WE CALL IT BEAR"],
                    "expected_label": ["correct"],
                }
            ).to_csv(manifest, index=False)
            pd.DataFrame(
                {
                    "utterance_id": ["u1", "u1"],
                    "word": ["WE", "BEAR"],
                    "target_phone": ["W", "R"],
                    "phone_group": ["glide", "rhotic"],
                    "duration_ms": [80, 40],
                    "prob_correct": [0.9, 0.8],
                    "decision": ["correct", "true_error"],
                }
            ).to_csv(output_dir / "u1_prediction.csv", index=False)
            pd.DataFrame({"word": ["WE"], "word_decision": ["correct"]}).to_csv(output_dir / "u1_word_summary.csv", index=False)
            report = evaluate_manifest(manifest, output_dir, run_predictions=False)
            self.assertEqual(report["predicted_true_error_count"], 1)
            self.assertTrue((output_dir / "false_positive_correct_audio.csv").exists())
            self.assertTrue((output_dir / "false_positive_analysis.md").exists())


if __name__ == "__main__":
    unittest.main()
