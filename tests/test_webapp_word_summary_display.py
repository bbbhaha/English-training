from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))

from app import apply_word_summary_display


def _prediction() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "word": ["AMERICA", "AMERICA"],
            "word_index": [1, 1],
            "target_phone": ["AH", "M"],
            "decision": ["correct", "correct"],
            "error_type": ["", ""],
            "alignment_quality": ["pass", "pass"],
        }
    )


def _summary(word_decision: str, error_type: str, possible_missing: bool) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "word": ["AMERICA"],
            "word_index": [1],
            "possible_missing_word": [possible_missing],
            "word_decision": [word_decision],
            "error_type": [error_type],
            "deletion_trigger_source": ["asr_missing_word" if possible_missing else "none"],
            "missing_word_reason": ["asr_missing_word" if possible_missing else ""],
            "alignment_quality": ["suspect" if possible_missing else "pass"],
        }
    )


class WebappWordSummaryDisplayTests(unittest.TestCase):
    def test_word_summary_deletion_overrides_phone_correct(self):
        display = apply_word_summary_display(
            _prediction(),
            _summary("true_error", "deletion", True),
            "deletion_only",
        )
        self.assertTrue(display["display_decision"].eq("漏读").all())
        self.assertTrue(display["display_error"].eq("漏读").all())
        self.assertTrue(display["display_align"].eq("suspect").all())
        self.assertTrue(display["display_error_type"].eq("deletion").all())
        self.assertTrue(display["error_type"].eq("").all())

    def test_word_summary_possible_deletion_overrides_phone_correct(self):
        display = apply_word_summary_display(
            _prediction(),
            _summary("uncertain_review", "possible_deletion", True),
            "deletion_only",
        )
        self.assertTrue(display["display_decision"].eq("疑似漏读/需复核").all())
        self.assertTrue(display["display_error"].eq("疑似漏读").all())
        self.assertTrue(display["display_align"].eq("suspect").all())
        self.assertTrue(display["display_error_type"].eq("possible_deletion").all())
        self.assertTrue(display["error_type"].eq("").all())

    def test_no_missing_word_displays_phone_correct(self):
        display = apply_word_summary_display(
            _prediction(),
            _summary("correct", "", False),
            "deletion_only",
        )
        self.assertTrue(display["display_decision"].eq("正确").all())
        self.assertTrue(display["display_error"].eq("0%").all())
        self.assertTrue(display["display_align"].eq("pass").all())


if __name__ == "__main__":
    unittest.main()
