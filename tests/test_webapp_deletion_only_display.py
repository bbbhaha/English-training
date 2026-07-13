from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))

from app import deletion_only_display_fields


class WebappDeletionOnlyDisplayTests(unittest.TestCase):
    def test_correct_row_with_error_probability_one_displays_clean(self):
        row = {
            "decision": "correct",
            "manual_calibrated_error_probability": 1.0,
            "alignment_quality": "pass",
            "error_type": "",
        }
        display = deletion_only_display_fields(row)
        self.assertEqual(display["error_display"], "0%")
        self.assertEqual(display["align_display"], "pass")
        self.assertEqual(display["decision_display"], "正确")
        self.assertNotEqual(display["error_display"], "100%")
        self.assertNotEqual(display["align_display"], "suspect")
        self.assertNotEqual(display["decision_display"], "需复核")

    def test_possible_deletion_display(self):
        row = {
            "decision": "uncertain_review",
            "manual_calibrated_error_probability": 1.0,
            "alignment_quality": "suspect",
            "error_type": "possible_deletion",
        }
        display = deletion_only_display_fields(row)
        self.assertEqual(display["error_display"], "疑似漏读")
        self.assertEqual(display["decision_display"], "疑似漏读/需复核")

    def test_frontend_uses_display_fields_not_probability(self):
        app_js = (ROOT / "webapp" / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("row.display_error", app_js)
        self.assertIn("row.display_decision", app_js)
        self.assertIn("row.display_align", app_js)
        self.assertIn("row.display_error_type", app_js)


if __name__ == "__main__":
    unittest.main()
