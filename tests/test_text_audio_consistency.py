from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pronunciation.deletion_detector import build_word_summary, detect_word_deletions
from pronunciation.text_audio_consistency import (
    check_text_audio_consistency,
    compare_target_with_asr,
    merge_consistency_into_phone_frame,
    merge_consistency_into_word_summary,
)


def _america_phone_frame(possible_missing: bool = True) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "word": ["MAKE"] + ["AMERICA"] * 7 + ["GREAT", "AGAIN"],
            "word_index": [0] + [1] * 7 + [2, 3],
            "target_phone": ["M", "AH", "M", "EH", "R", "IH", "K", "AH", "G", "AH"],
            "phone_index": list(range(10)),
            "start_ms": [0, 100, 230, 390, 720, 770, 820, 870, 1100, 1300],
            "end_ms": [80, 230, 390, 720, 770, 820, 870, 1090, 1220, 1450],
            "duration_ms": [80, 130, 160, 330, 50, 50, 50, 220, 120, 150],
            "alignment_quality": ["pass"] * 10,
            "manual_calibrated_error_probability": [0.1] + ([0.99] * 7 if possible_missing else [0.1] * 7) + [0.1, 0.1],
            "prob_correct": [0.9] * 10,
            "decision": ["correct"] * 10,
            "review_reason": [""] * 10,
        }
    )
    detected, _ = detect_word_deletions(frame)
    return detected


class TextAudioConsistencyTests(unittest.TestCase):
    def test_compare_marks_all_missing_target_words(self):
        result = compare_target_with_asr("SHE SEES THE BLUE BIRD", "SHE THE BIRD")
        missing = result.loc[result["asr_missing_word"], "word"].tolist()
        self.assertEqual(missing, ["SEES", "BLUE"])
        self.assertTrue(result.loc[result["word"].isin(missing), "text_audio_mismatch_score"].eq(0.95).all())

    def test_compare_marks_replaced_word(self):
        result = compare_target_with_asr("SHE SEES BLUE", "SHE LIKES BLUE")
        replaced = result[result["word"].eq("SEES")].iloc[0]
        self.assertEqual(replaced["asr_edit_op"], "replace")
        self.assertTrue(bool(replaced["asr_substituted_word"]))
        self.assertEqual(replaced["text_audio_mismatch_score"], 0.85)

    def test_asr_transcript_deletion_marks_america_missing(self):
        consistency, meta = check_text_audio_consistency(
            target_text="MAKE AMERICA GREAT AGAIN",
            asr_transcript="MAKE GREAT AGAIN",
        )
        america = consistency[consistency["target_word"].eq("AMERICA")].iloc[0]
        self.assertEqual(america["asr_word_status"], "deletion")
        self.assertTrue(bool(america["asr_missing_word"]))
        self.assertEqual(meta["asr_transcript_normalized"], "MAKE GREAT AGAIN")

    def test_asr_deletion_promotes_word_summary_to_true_error(self):
        consistency, meta = check_text_audio_consistency(
            target_text="MAKE AMERICA GREAT AGAIN",
            asr_transcript="MAKE GREAT AGAIN",
        )
        phone_frame = merge_consistency_into_phone_frame(
            _america_phone_frame(possible_missing=True),
            consistency,
            asr_transcript=meta["asr_transcript"],
        )
        summary = build_word_summary(phone_frame)
        summary = merge_consistency_into_word_summary(summary, consistency, asr_transcript=meta["asr_transcript"])
        america = summary[summary["word"].eq("AMERICA")].iloc[0]
        self.assertEqual(america["error_type"], "deletion")
        self.assertEqual(america["word_decision"], "true_error")
        self.assertEqual(america["deletion_confidence"], "high")

    def test_matching_asr_transcript_does_not_mark_deletion(self):
        consistency, meta = check_text_audio_consistency(
            target_text="MAKE AMERICA GREAT AGAIN",
            asr_transcript="MAKE AMERICA GREAT AGAIN",
        )
        america = consistency[consistency["target_word"].eq("AMERICA")].iloc[0]
        self.assertEqual(america["asr_word_status"], "match")
        self.assertFalse(bool(america["asr_missing_word"]))

        phone_frame = merge_consistency_into_phone_frame(
            _america_phone_frame(possible_missing=False),
            consistency,
            asr_transcript=meta["asr_transcript"],
        )
        summary = build_word_summary(phone_frame)
        america_summary = summary[summary["word"].eq("AMERICA")].iloc[0]
        self.assertNotEqual(america_summary["error_type"], "deletion")
        self.assertNotEqual(america_summary["word_decision"], "true_error")


if __name__ == "__main__":
    unittest.main()
