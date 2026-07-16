from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))

from app import apply_word_summary_display, format_final_error_display


class FrontendErrorPercentDisplayTests(unittest.TestCase):
    def test_probability_formatting(self):
        self.assertEqual(format_final_error_display(0.95, "deletion"), "95% 漏读")
        self.assertEqual(format_final_error_display(0.85, "text_audio_mismatch"), "85% 文本音频不一致")
        self.assertEqual(format_final_error_display(0.0, ""), "0%")

    def test_summary_probability_controls_frontend(self):
        prediction = pd.DataFrame([{
            "word": "BLUE", "word_index": 3, "target_phone": "B", "decision": "correct",
            "error_type": "", "alignment_quality": "pass",
        }])
        summary = pd.DataFrame([{
            "word": "BLUE", "word_index": 3, "possible_missing_word": False,
            "word_decision": "true_error", "error_type": "text_audio_mismatch",
            "alignment_quality": "pass", "final_word_decision": "substituted_word",
            "final_error_type": "text_audio_mismatch", "final_error_probability": 0.85,
            "evidence_summary": "target word replaced in ASR transcript",
        }])
        display = apply_word_summary_display(prediction, summary)
        self.assertEqual(display.iloc[0]["display_error"], "85% 文本音频不一致")
        self.assertEqual(display.iloc[0]["display_decision"], "文本音频不一致")


if __name__ == "__main__":
    unittest.main()
