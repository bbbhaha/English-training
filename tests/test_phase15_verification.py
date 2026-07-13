from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase15_verification.attribute_verifier import apply_attribute_verifier
from phase15_verification.analysis import add_evidence_columns, evidence_pattern_metrics
from phase15_verification.decision_aggregator import aggregate_decisions
from phase15_verification.labels import infer_error_labels, main_error_score
from phase15_verification.phone_attributes import load_phone_attributes


class Phase15VerificationTests(unittest.TestCase):
    def test_phase1_gold_binary_means_zero_is_error(self):
        frame = pd.DataFrame({"gold_binary": [0, 1, 1, 0]})
        self.assertEqual(infer_error_labels(frame, "gold_binary").tolist(), [1, 0, 0, 1])

    def test_prob_correct_is_inverted_to_error_score(self):
        frame = pd.DataFrame({"prob_correct": [0.9, 0.2]})
        self.assertEqual(main_error_score(frame).round(3).tolist(), [0.1, 0.8])

    def test_attribute_and_aggregator_outputs(self):
        mapping = load_phone_attributes(ROOT / "configs/phase15/phone_attributes.json")
        frame = pd.DataFrame(
            {
                "utterance_id": ["u1", "u1"],
                "word": ["VERY", "VERY"],
                "target_phone": ["V", "IY"],
                "phone_index": [0, 1],
                "duration_ms": [20, 150],
                "prob_correct": [0.1, 0.9],
                "prediction": [0, 1],
                "gold_binary": [0, 1],
            }
        )
        out = apply_attribute_verifier(frame, mapping)
        out["retrieval_verifier_decision"] = ["likely_error", "likely_correct"]
        out["oneclass_verifier_decision"] = ["high_confidence_error", "likely_correct"]
        out["proto_margin"] = [-0.2, 0.2]
        out["oneclass_anomaly_score"] = [0.8, 0.1]
        out = aggregate_decisions(out, {"aggregator": {"min_support_votes": 2}})
        self.assertIn("final_decision", out.columns)
        self.assertEqual(out.loc[0, "final_decision"], "high_confidence_error")

    def test_evidence_pattern_metrics(self):
        frame = pd.DataFrame(
            {
                "target_phone": ["R", "R"],
                "gold_binary": [0, 1],
                "prob_correct": [0.1, 0.9],
                "prediction": [0, 1],
                "attribute_risk_score": [0.8, 0.1],
                "proto_margin": [-0.2, 0.3],
                "oneclass_anomaly_score": [0.7, 0.1],
                "final_decision": ["high_confidence_error", "correct"],
                "final_error_score": [0.8, 0.1],
            }
        )
        out = add_evidence_columns(frame, {"attribute": {"high_risk_threshold": 0.5}})
        self.assertIn("evidence_pattern", out.columns)
        metrics = evidence_pattern_metrics(out, "gold_binary")
        self.assertGreaterEqual(len(metrics), 1)

    def test_strict_consensus_downgrades_weak_evidence(self):
        frame = pd.DataFrame(
            {
                "target_phone": ["R"],
                "phone_group": ["liquid"],
                "prob_correct": [0.1],
                "prediction": [0],
                "attribute_risk_score": [0.1],
                "proto_margin": [0.3],
                "oneclass_anomaly_score": [0.1],
            }
        )
        out = aggregate_decisions(
            frame,
            {
                "aggregator": {"aggregator_mode": "strict_consensus", "min_main_error_score": 0.3},
                "strict_consensus": {"min_evidence_count": 2, "enabled_phone_groups": ["liquid"]},
            },
        )
        self.assertEqual(out.loc[0, "final_decision"], "uncertain_review")


if __name__ == "__main__":
    unittest.main()
