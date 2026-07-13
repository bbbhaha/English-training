from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "webapp"))

from app import apply_word_summary_display, deletion_only_display_fields
from pronunciation.decision import apply_deletion_only_override


def prediction(alignment_quality="pass", deletion_trigger_source="none") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "word": "AMERICA",
                "word_index": 1,
                "target_phone": "AH",
                "alignment_quality": alignment_quality,
                "deletion_trigger_source": deletion_trigger_source,
                "possible_missing_word": deletion_trigger_source != "none",
                "missing_word_reason": "asr_missing_word" if deletion_trigger_source != "none" else "",
                "decision": "correct",
                "error_type": "",
                "review_reason": "",
                "manual_calibrated_error_probability": 1.0,
                "model_error_score": 1.0,
                "confidence": 1.0,
            }
        ]
    )


def summary(alignment_quality="pass", error_type="", possible_missing_word=False) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "word": "AMERICA",
                "word_index": 1,
                "possible_missing_word": possible_missing_word,
                "word_decision": "true_error" if error_type == "deletion" else "correct",
                "error_type": error_type,
                "deletion_trigger_source": "asr_missing_word" if possible_missing_word else "none",
                "missing_word_reason": "asr_missing_word" if possible_missing_word else "",
                "alignment_quality": alignment_quality,
            }
        ]
    )


class DeletionOnlyAlignmentIssueTests(unittest.TestCase):
    def test_bad_phone_alignment_is_uncertain_review(self):
        out, _ = apply_deletion_only_override(prediction("bad"))
        row = out.iloc[0]

        self.assertEqual(row["decision"], "uncertain_review")
        self.assertEqual(row["error_type"], "alignment_issue")
        self.assertEqual(row["review_reason"], "bad_alignment")
        self.assertEqual(row["manual_calibrated_error_probability"], 0.0)
        self.assertEqual(row["model_error_score"], 0.0)
        self.assertEqual(row["confidence"], 0.0)
        self.assertEqual(row["alignment_quality"], "bad")
        self.assertEqual(row["original_alignment_quality"], "bad")

    def test_bad_word_alignment_is_uncertain_review(self):
        out, word_summary = apply_deletion_only_override(prediction("pass"), summary("bad"))

        self.assertEqual(out.loc[0, "decision"], "uncertain_review")
        self.assertEqual(out.loc[0, "error_type"], "alignment_issue")
        self.assertEqual(word_summary.loc[0, "word_decision"], "uncertain_review")
        self.assertEqual(word_summary.loc[0, "error_type"], "alignment_issue")
        self.assertEqual(word_summary.loc[0, "missing_word_reason"], "bad_alignment")
        self.assertEqual(word_summary.loc[0, "alignment_quality"], "bad")
        self.assertFalse(bool(word_summary.loc[0, "possible_missing_word"]))

    def test_any_bad_phone_marks_word_summary_bad(self):
        out, word_summary = apply_deletion_only_override(prediction("bad"), summary("pass"))

        self.assertEqual(out.loc[0, "decision"], "uncertain_review")
        self.assertEqual(word_summary.loc[0, "word_decision"], "uncertain_review")
        self.assertEqual(word_summary.loc[0, "error_type"], "alignment_issue")
        self.assertEqual(word_summary.loc[0, "alignment_quality"], "bad")

    def test_good_alignment_without_deletion_is_correct(self):
        for quality in ["pass", "good"]:
            with self.subTest(quality=quality):
                out, _ = apply_deletion_only_override(prediction(quality))
                self.assertEqual(out.loc[0, "decision"], "correct")
                self.assertEqual(out.loc[0, "error_type"], "")
                self.assertEqual(out.loc[0, "confidence"], 1.0)

    def test_deletion_display_has_priority_over_bad_alignment(self):
        pred = prediction("bad")
        word_summary = summary("bad", "deletion", True)
        display = apply_word_summary_display(pred, word_summary)

        self.assertEqual(display.loc[0, "display_decision"], "漏读")
        self.assertEqual(display.loc[0, "display_error"], "漏读")
        self.assertEqual(display.loc[0, "display_align"], "suspect")
        self.assertEqual(display.loc[0, "display_error_type"], "deletion")

    def test_alignment_issue_frontend_display(self):
        pred = prediction("bad")
        pred.loc[0, "error_type"] = "alignment_issue"
        word_summary = summary("bad", "alignment_issue", False)
        display = apply_word_summary_display(pred, word_summary)

        self.assertEqual(display.loc[0, "display_decision"], "需复核")
        self.assertEqual(display.loc[0, "display_error"], "对齐失败")
        self.assertEqual(display.loc[0, "display_align"], "bad")
        self.assertEqual(display.loc[0, "display_error_type"], "alignment_issue")

        legacy_display = deletion_only_display_fields(
            {"decision": "correct", "error_type": "", "alignment_quality": "bad"}
        )
        self.assertEqual(legacy_display["decision_display"], "需复核")
        self.assertEqual(legacy_display["error_display"], "对齐失败")
        self.assertEqual(legacy_display["align_display"], "bad")

        precomputed_display = deletion_only_display_fields(
            {
                "display_decision": "正确",
                "display_error": "0%",
                "display_align": "bad",
                "display_error_type": "",
            }
        )
        self.assertEqual(precomputed_display["decision_display"], "需复核")
        self.assertEqual(precomputed_display["error_display"], "对齐失败")

    def test_failed_alignment_is_normalized_to_bad(self):
        for quality in ["failed", "alignment_failed"]:
            with self.subTest(quality=quality):
                out, _ = apply_deletion_only_override(prediction(quality))
                self.assertEqual(out.loc[0, "decision"], "uncertain_review")
                self.assertEqual(out.loc[0, "alignment_quality"], "bad")

    def test_temporary_fake_display_control_is_removed(self):
        index_html = (ROOT / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
        app_js = (ROOT / "webapp" / "static" / "app.js").read_text(encoding="utf-8")
        self.assertNotIn("测试漏读显示", index_html)
        self.assertNotIn("debugDeletionBtn", app_js)
        self.assertNotIn("debug-fake-deletion", app_js)


if __name__ == "__main__":
    unittest.main()
