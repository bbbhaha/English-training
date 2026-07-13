import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pronunciation.mandarin_confusion_prior import classify_mandarin_confusion


class MandarinConfusionPriorTests(unittest.TestCase):
    def test_th_to_s_is_common_mandarin_pattern(self):
        result = classify_mandarin_confusion("TH", "S", "initial")
        self.assertEqual(result["confusion_type"], "th_fronting_or_stopping")
        self.assertEqual(result["severity"], "medium")
        self.assertTrue(result["is_common_mandarin_error"])

    def test_ih_iy_is_low_severity(self):
        result = classify_mandarin_confusion("IH", "IY", "medial")
        self.assertEqual(result["confusion_type"], "ih_iy_confusion")
        self.assertEqual(result["severity"], "low")
        self.assertFalse(result["is_likely_intelligibility_error"])

    def test_final_consonant_deletion_is_high_severity(self):
        result = classify_mandarin_confusion("T", "", "final")
        self.assertEqual(result["confusion_type"], "final_consonant_deletion")
        self.assertTrue(result["is_likely_intelligibility_error"])


if __name__ == "__main__":
    unittest.main()
