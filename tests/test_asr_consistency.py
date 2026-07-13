import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pronunciation.asr_consistency import compare_target_with_asr


class AsrConsistencyTests(unittest.TestCase):
    def test_missing_america_is_a_deletion(self):
        result = compare_target_with_asr("MAKE AMERICA GREAT AGAIN", "MAKE GREAT AGAIN")
        america = result[result["word"].eq("AMERICA")].iloc[0]
        self.assertEqual(america["asr_word_status"], "deletion")
        self.assertEqual(america["asr_edit_op"], "delete")
        self.assertTrue(bool(america["asr_missing_word"]))

    def test_complete_transcript_has_no_missing_word(self):
        result = compare_target_with_asr("MAKE AMERICA GREAT AGAIN", "MAKE AMERICA GREAT AGAIN")
        self.assertTrue(result["asr_word_status"].eq("match").all())
        self.assertFalse(result["asr_missing_word"].any())


if __name__ == "__main__":
    unittest.main()
