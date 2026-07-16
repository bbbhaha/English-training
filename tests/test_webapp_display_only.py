from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))

from app import apply_word_summary_display


def prediction_frame(word_index=1) -> pd.DataFrame:
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


def summary_frame(error_type: str, possible_missing_word: bool, word_index=1) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "word": "AMERICA",
                "word_index": word_index,
                "possible_missing_word": possible_missing_word,
                "word_decision": "true_error" if error_type == "deletion" else (
                    "uncertain_review" if error_type == "possible_deletion" else "correct"
                ),
                "error_type": error_type,
                "deletion_trigger_source": "asr_missing_word" if possible_missing_word else "none",
                "missing_word_reason": "missing_in_asr_transcript" if possible_missing_word else "",
                "alignment_quality": "suspect" if possible_missing_word else "pass",
            }
        ]
    )


class WebappDisplayOnlyTests(unittest.TestCase):
    def test_deletion_overrides_phone_correct_for_display(self):
        display = apply_word_summary_display(prediction_frame(), summary_frame("deletion", True))
        self.assertEqual(display.loc[0, "display_decision"], "漏读")
        self.assertEqual(display.loc[0, "display_error_type"], "deletion")
        self.assertEqual(display.loc[0, "error_type"], "")

    def test_possible_deletion_overrides_phone_correct_for_display(self):
        display = apply_word_summary_display(prediction_frame(), summary_frame("possible_deletion", True))
        self.assertEqual(display.loc[0, "display_decision"], "疑似漏读/需复核")
        self.assertEqual(display.loc[0, "display_error_type"], "possible_deletion")

    def test_non_missing_word_displays_correct(self):
        display = apply_word_summary_display(prediction_frame(), summary_frame("", False))
        self.assertEqual(display.loc[0, "display_decision"], "正确")
        self.assertEqual(display.loc[0, "display_error"], "0%")
        self.assertEqual(display.loc[0, "display_align"], "pass")

    def test_int_and_string_word_index_still_merge(self):
        display = apply_word_summary_display(prediction_frame(1), summary_frame("deletion", True, "1"))
        self.assertEqual(display.loc[0, "display_decision"], "漏读")

    def test_static_table_uses_only_display_fields(self):
        app_js = (ROOT / "webapp" / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("row.display_decision", app_js)
        self.assertIn("row.display_error", app_js)
        self.assertIn("row.display_align", app_js)
        self.assertIn("row.display_error_type", app_js)
        self.assertNotIn("row.manual_calibrated_error_probability", app_js)

    def test_g2p_issue_is_displayed_as_word_not_collected(self):
        prediction = prediction_frame()
        prediction["lexicon_status"] = "failed"
        prediction["g2p_source"] = "failed"
        prediction["g2p_confidence"] = "low"
        summary = summary_frame("g2p_issue", False)
        summary["lexicon_status"] = "failed"
        summary["g2p_source"] = "failed"
        summary["g2p_confidence"] = "low"
        display = apply_word_summary_display(prediction, summary)
        self.assertEqual(display.loc[0, "display_decision"], "单词暂未收录")
        self.assertEqual(display.loc[0, "display_error"], "无法判断")
        self.assertEqual(display.loc[0, "display_error_type"], "g2p_issue")


if __name__ == "__main__":
    unittest.main()
