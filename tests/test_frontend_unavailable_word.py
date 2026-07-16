from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "webapp"))

from app import apply_phone_diagnosis_display
from pronunciation.phone_decision import apply_phone_decisions


class FrontendUnavailableWordTests(unittest.TestCase):
    def test_failed_g2p_displays_word_not_in_collection(self):
        prediction = pd.DataFrame([{
            "word": "ZZZXQ",
            "word_index": 0,
            "target_phone": "<UNK>",
            "g2p_status": "failed",
            "lexicon_status": "failed",
            "alignment_quality": "bad",
            "model_error_score": 0.99,
        }])
        diagnosed = apply_phone_decisions(prediction)
        display = apply_phone_diagnosis_display(diagnosed)
        self.assertEqual(display.loc[0, "display_decision"], "单词暂未收录")
        self.assertEqual(display.loc[0, "display_error"], "无法判断")
        self.assertEqual(display.loc[0, "display_error_type"], "g2p_issue")

    def test_word_deletion_overrides_phone_correct(self):
        prediction = pd.DataFrame([{
            "word": "AMERICA",
            "word_index": 1,
            "target_phone": "AH",
            "phone_decision": "correct",
            "phone_error_type": "",
            "phone_error_percent": 0.0,
            "alignment_quality": "pass",
            "g2p_status": "success",
            "lexicon_status": "cmudict",
        }])
        summary = pd.DataFrame([{
            "word_index": 1,
            "deletion_decision": "deletion",
            "deletion_score": 0.93,
            "final_word_decision": "deletion",
            "final_error_type": "deletion",
            "alignment_quality": "suspect",
        }])
        display = apply_phone_diagnosis_display(prediction, summary)
        self.assertEqual(display.loc[0, "display_decision"], "漏读")
        self.assertEqual(display.loc[0, "display_align"], "suspect")


if __name__ == "__main__":
    unittest.main()
