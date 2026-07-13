import unittest
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pronunciation.word_deletion_model import word_deletion_detector


class WordDeletionModelTests(unittest.TestCase):
    def _summary(self, duration=400.0, alignment="pass"):
        return pd.DataFrame([{
            "word": "AMERICA", "word_index": 1, "phone_count": 7,
            "word_duration_ms": duration, "alignment_quality": alignment,
        }])

    def test_asr_missing_and_short_is_deletion(self):
        asr = pd.DataFrame([{"word": "AMERICA", "word_index": 1, "asr_missing_word": True}])
        result = word_deletion_detector(self._summary(150.0), asr)
        self.assertEqual(result.iloc[0]["deletion_decision"], "deletion")

    def test_asr_missing_without_shortness_is_possible_deletion(self):
        asr = pd.DataFrame([{"word": "AMERICA", "word_index": 1, "asr_missing_word": True}])
        result = word_deletion_detector(self._summary(400.0), asr)
        self.assertEqual(result.iloc[0]["deletion_decision"], "possible_deletion")

    def test_bad_alignment_is_alignment_issue(self):
        result = word_deletion_detector(self._summary(400.0, "bad"))
        self.assertEqual(result.iloc[0]["deletion_decision"], "alignment_issue")

    def test_normal_complete_word_is_correct(self):
        result = word_deletion_detector(self._summary())
        self.assertEqual(result.iloc[0]["deletion_decision"], "correct")


if __name__ == "__main__":
    unittest.main()
