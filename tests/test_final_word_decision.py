import unittest
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pronunciation.final_word_decision import fuse_word_decisions, run_word_level_diagnosis
from pronunciation.word_mispronunciation_model import word_mispronunciation_detector


class FinalWordDecisionTests(unittest.TestCase):
    def test_deletion_has_priority_over_mispronunciation(self):
        frame = pd.DataFrame([{
            "deletion_decision": "deletion",
            "mispronunciation_decision": "mispronounced",
        }])
        self.assertEqual(fuse_word_decisions(frame).iloc[0]["final_word_decision"], "deletion")

    def test_alignment_issue_has_priority(self):
        frame = pd.DataFrame([{
            "deletion_decision": "alignment_issue",
            "mispronunciation_decision": "mispronounced",
        }])
        self.assertEqual(fuse_word_decisions(frame).iloc[0]["final_word_decision"], "alignment_issue")

    def test_failed_g2p_is_review_not_true_error(self):
        frame = pd.DataFrame([{
            "lexicon_status": "failed",
            "deletion_decision": "alignment_issue",
            "mispronunciation_decision": "mispronounced",
        }])
        result = fuse_word_decisions(frame).iloc[0]
        self.assertEqual(result["final_word_decision"], "g2p_issue")
        self.assertEqual(result["final_error_type"], "g2p_issue")

    def test_america_manual_asr_deletion_reaches_final_output(self):
        phones = pd.DataFrame([
            {"word": "AMERICA", "word_index": 1, "target_phone": phone, "duration_ms": 20.0,
             "alignment_quality": "pass"}
            for phone in ["AH", "M", "EH", "R", "IH", "K", "AH"]
        ])
        summary = pd.DataFrame([{
            "word": "AMERICA", "word_index": 1, "phone_count": 7,
            "word_duration_ms": 140.0, "alignment_quality": "pass",
        }])
        asr = pd.DataFrame([{
            "word": "AMERICA", "word_index": 1, "asr_missing_word": True,
            "asr_word_status": "deletion", "asr_edit_op": "delete",
        }])
        result = run_word_level_diagnosis(phones, summary, asr)
        self.assertEqual(result.iloc[0]["final_word_decision"], "deletion")

    def test_complete_words_with_good_alignment_are_correct(self):
        phones = pd.DataFrame([
            {"word": "MAKE", "word_index": 0, "target_phone": "M", "duration_ms": 80.0,
             "alignment_quality": "pass"},
            {"word": "MAKE", "word_index": 0, "target_phone": "EY", "duration_ms": 120.0,
             "alignment_quality": "pass"},
            {"word": "GREAT", "word_index": 1, "target_phone": "G", "duration_ms": 100.0,
             "alignment_quality": "pass"},
            {"word": "GREAT", "word_index": 1, "target_phone": "EY", "duration_ms": 150.0,
             "alignment_quality": "pass"},
        ])
        summary = pd.DataFrame([
            {"word": "MAKE", "word_index": 0, "phone_count": 2, "word_duration_ms": 200.0,
             "alignment_quality": "pass"},
            {"word": "GREAT", "word_index": 1, "phone_count": 2, "word_duration_ms": 250.0,
             "alignment_quality": "pass"},
        ])
        result = run_word_level_diagnosis(phones, summary)
        self.assertTrue(result["final_word_decision"].eq("correct").all())

    def test_th_to_s_can_be_mispronounced_with_strong_acoustic_evidence(self):
        frame = pd.DataFrame([{
            "target_phone_seq": "TH IH S",
            "predicted_phone_seq": "S IH S",
            "avg_phone_score": 0.2,
            "min_phone_score": 0.1,
            "alignment_quality": "pass",
            "asr_edit_op": "substitute",
        }])
        result = word_mispronunciation_detector(frame).iloc[0]
        self.assertEqual(result["mispronunciation_decision"], "mispronounced")

    def test_ih_to_iy_is_not_high_confidence_mispronunciation(self):
        frame = pd.DataFrame([{
            "target_phone_seq": "IH",
            "predicted_phone_seq": "IY",
            "avg_phone_score": 0.8,
            "min_phone_score": 0.8,
            "alignment_quality": "pass",
            "asr_edit_op": "match",
        }])
        decision = word_mispronunciation_detector(frame).iloc[0]["mispronunciation_decision"]
        self.assertIn(decision, {"acceptable_accent", "uncertain_review"})


if __name__ == "__main__":
    unittest.main()
