from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))

from app import apply_word_summary_display, deletion_only_display_fields


class WebappDisplayDeletionOnlyTests(unittest.TestCase):
    def test_correct_row_never_displays_error_probability(self):
        display = deletion_only_display_fields(
            {
                "decision": "correct",
                "manual_calibrated_error_probability": 1.0,
                "alignment_quality": "pass",
                "error_type": "",
            }
        )
        self.assertEqual(display["error_display"], "0%")
        self.assertEqual(display["align_display"], "pass")
        self.assertEqual(display["decision_display"], "正确")
        self.assertNotEqual(display["error_display"], "100%")
        self.assertNotEqual(display["align_display"], "suspect")
        self.assertNotEqual(display["decision_display"], "需复核")

    @staticmethod
    def _prediction(word_index=1):
        return pd.DataFrame(
            [
                {
                    "word": "AMERICA",
                    "word_index": word_index,
                    "target_phone": "AH",
                    "decision": "correct",
                    "error_type": "",
                    "alignment_quality": "pass",
                    "manual_calibrated_error_probability": 0.0,
                }
            ]
        )

    @staticmethod
    def _summary(error_type, possible_missing_word, word_index=1):
        return pd.DataFrame(
            [
                {
                    "word_index": word_index,
                    "possible_missing_word": possible_missing_word,
                    "word_decision": "true_error" if error_type == "deletion" else "uncertain_review",
                    "error_type": error_type,
                    "deletion_trigger_source": "asr_missing_word" if possible_missing_word else "none",
                    "missing_word_reason": "missing_in_asr_transcript" if possible_missing_word else "",
                    "alignment_quality": "suspect" if possible_missing_word else "pass",
                }
            ]
        )

    def test_word_summary_deletion_has_display_priority(self):
        display = apply_word_summary_display(self._prediction(), self._summary("deletion", True))
        self.assertEqual(display.loc[0, "display_decision"], "漏读")
        self.assertEqual(display.loc[0, "display_error"], "漏读")
        self.assertEqual(display.loc[0, "display_align"], "suspect")

    def test_word_summary_possible_deletion_has_display_priority(self):
        display = apply_word_summary_display(self._prediction(), self._summary("possible_deletion", True))
        self.assertEqual(display.loc[0, "display_decision"], "疑似漏读/需复核")
        self.assertEqual(display.loc[0, "display_error"], "疑似漏读")

    def test_non_missing_word_displays_correct(self):
        display = apply_word_summary_display(self._prediction(), self._summary("", False))
        self.assertEqual(display.loc[0, "display_decision"], "正确")
        self.assertEqual(display.loc[0, "display_error"], "0%")

    def test_int_and_string_word_index_merge(self):
        display = apply_word_summary_display(self._prediction(1), self._summary("deletion", True, "1"))
        self.assertEqual(display.loc[0, "display_decision"], "漏读")

    def test_possible_missing_word_is_fallback_when_error_type_is_empty(self):
        display = apply_word_summary_display(self._prediction(), self._summary("", True))
        self.assertEqual(display.loc[0, "display_decision"], "疑似漏读/需复核")
        self.assertEqual(display.loc[0, "display_error"], "疑似漏读")
        self.assertEqual(display.loc[0, "display_error_type"], "possible_deletion")


if __name__ == "__main__":
    unittest.main()
