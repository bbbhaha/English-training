from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pronunciation.final_word_decision import compute_final_word_decision


class FinalErrorProbabilityTests(unittest.TestCase):
    def test_asr_missing_word_has_high_deletion_probability(self):
        result = compute_final_word_decision(pd.Series({
            "asr_missing_word": True,
            "text_audio_mismatch_score": 0.95,
        }))
        self.assertEqual(result["final_word_decision"], "deletion")
        self.assertEqual(result["final_error_type"], "deletion")
        self.assertEqual(result["final_error_percent"], 95)

    def test_asr_substitution_is_text_audio_mismatch(self):
        result = compute_final_word_decision(pd.Series({
            "asr_substituted_word": True,
            "text_audio_mismatch_score": 0.85,
        }))
        self.assertEqual(result["final_word_decision"], "substituted_word")
        self.assertEqual(result["final_error_type"], "text_audio_mismatch")
        self.assertEqual(result["final_error_percent"], 85)

    def test_bad_alignment_is_review_not_certain_error(self):
        result = compute_final_word_decision(pd.Series({"alignment_quality": "bad"}))
        self.assertEqual(result["final_word_decision"], "alignment_issue")
        self.assertEqual(result["final_decision"], "uncertain_review")
        self.assertEqual(result["final_error_probability"], 0.5)

    def test_clean_word_has_zero_error_probability(self):
        result = compute_final_word_decision(pd.Series({"alignment_quality": "pass"}))
        self.assertEqual(result["final_word_decision"], "correct")
        self.assertEqual(result["final_error_probability"], 0.0)


if __name__ == "__main__":
    unittest.main()
