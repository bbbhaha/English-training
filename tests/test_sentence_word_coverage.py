from argparse import Namespace
from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "webapp"))

from pronunciation.alignment import ensure_alignment_coverage
from pronunciation.g2p import text_to_phones
from pronunciation.target_words import build_target_word_table, ensure_word_summary_coverage
from predict_pronunciation import ensure_prediction_coverage
from app import apply_word_summary_display


TEXT = "SHE SEES THE BLUE BIRD"
WORDS = ["SHE", "SEES", "THE", "BLUE", "BIRD"]


class SentenceWordCoverageTests(unittest.TestCase):
    def setUp(self):
        self.target = build_target_word_table(TEXT, utterance_id="coverage")
        self.g2p = text_to_phones(TEXT, target_word_table=self.target)
        self.g2p_frame = pd.DataFrame(self.g2p.phones)

    def test_target_word_table_has_all_five_words(self):
        self.assertEqual(self.target["word"].tolist(), WORDS)
        self.assertEqual(self.target["word_index"].tolist(), list(range(5)))

    def test_g2p_covers_all_target_words(self):
        self.assertEqual(set(self.g2p_frame["word_index"]), set(range(5)))
        self.assertFalse(self.g2p_frame.groupby("word_index")["target_phone"].count().eq(0).any())

    def test_alignment_coverage_restores_missing_words(self):
        partial = self.g2p_frame[self.g2p_frame["word_index"].isin({0, 2})].copy()
        partial["start_ms"] = range(len(partial))
        partial["end_ms"] = partial["start_ms"] + 50
        partial["duration_ms"] = 50
        partial["alignment_quality"] = "pass"
        partial["review_reason"] = ""
        result = ensure_alignment_coverage(self.g2p_frame, partial)
        self.assertEqual(set(result["word_index"]), set(range(5)))
        restored = result[result["word_index"].isin({1, 3, 4})]
        self.assertTrue(restored["alignment_quality"].eq("bad").all())
        self.assertTrue(restored["review_reason"].eq("alignment_missing").all())

    def test_prediction_coverage_restores_missing_words(self):
        partial = self.g2p_frame[self.g2p_frame["word_index"].isin({0, 2})].copy()
        partial["decision"] = "correct"
        partial["alignment_quality"] = "pass"
        result = ensure_prediction_coverage(partial, self.target, self.g2p)
        self.assertEqual(set(result["word_index"]), set(range(5)))
        restored = result[result["word_index"].isin({1, 3, 4})]
        self.assertTrue(restored["decision"].eq("uncertain_review").all())
        self.assertTrue(restored["review_reason"].eq("missing_from_prediction_pipeline").all())

    def test_word_summary_coverage_has_all_five_words(self):
        partial = pd.DataFrame([
            {"word_index": 0, "word": "SHE", "word_decision": "correct", "alignment_quality": "pass"},
            {"word_index": 2, "word": "THE", "word_decision": "correct", "alignment_quality": "pass"},
        ])
        result = ensure_word_summary_coverage(self.target, partial)
        self.assertEqual(result["word"].tolist(), WORDS)
        restored = result[result["word_index"].isin({1, 3, 4})]
        self.assertTrue(restored["word_decision"].eq("uncertain_review").all())

    def test_frontend_display_uses_word_summary_as_master(self):
        prediction = self.g2p_frame[self.g2p_frame["word_index"].isin({0, 2})].copy()
        prediction["decision"] = "correct"
        prediction["error_type"] = ""
        prediction["alignment_quality"] = "pass"
        summary = ensure_word_summary_coverage(self.target, pd.DataFrame([
            {"word_index": 0, "word_decision": "correct", "error_type": "", "alignment_quality": "pass"},
            {"word_index": 2, "word_decision": "correct", "error_type": "", "alignment_quality": "pass"},
        ]))
        display = apply_word_summary_display(prediction, summary, "deletion_only")
        self.assertEqual(display["word"].tolist(), WORDS)
        self.assertEqual(len(display), 5)


if __name__ == "__main__":
    unittest.main()
