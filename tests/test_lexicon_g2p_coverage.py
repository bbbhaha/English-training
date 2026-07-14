from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pronunciation.g2p import text_to_phones
from pronunciation.lexicon import get_best_pronunciation
from pronunciation.word_mispronunciation_model import word_mispronunciation_detector


class LexiconG2PCoverageTests(unittest.TestCase):
    def test_sentence_covers_all_five_words(self):
        result = text_to_phones("SHE SEES THE BLUE BIRD")
        self.assertEqual([word["word"] for word in result.words], ["SHE", "SEES", "THE", "BLUE", "BIRD"])
        self.assertEqual({row["word_index"] for row in result.phones}, set(range(5)))

    def test_sees_blue_and_bird_are_not_unknown(self):
        result = text_to_phones("SEES BLUE BIRD")
        for word in result.words:
            self.assertNotIn("<UNK>", word["phones"])
            self.assertNotEqual(word["lexicon_status"], "failed")

    def test_unknown_word_is_kept_and_never_true_error(self):
        result = text_to_phones("XYZABC")
        word = result.words[0]
        self.assertEqual(word["word"], "XYZABC")
        self.assertIn(word["lexicon_status"], {"failed", "g2p_en", "phonemizer"})
        if word["lexicon_status"] == "failed":
            self.assertEqual(result.phones[0]["target_phone"], "<UNK>")
            self.assertEqual(result.phones[0]["error_type"], "g2p_issue")
            self.assertEqual(result.phones[0]["decision"], "uncertain_review")

    def test_the_has_two_pronunciation_variants(self):
        result = get_best_pronunciation("THE")
        self.assertIn(["DH", "AH"], result["pronunciations"])
        self.assertIn(["DH", "IY"], result["pronunciations"])
        self.assertGreaterEqual(result["num_pronunciation_variants"], 2)

    def test_inferred_unseen_pronunciation_is_conservative(self):
        features = pd.DataFrame([{
            "word": "EXAMPLE",
            "target_phone_seq": "IH G Z AE M P AH L",
            "predicted_phone_seq": "EH G Z AE M B AH L",
            "avg_phone_score": 0.1,
            "min_phone_score": 0.05,
            "alignment_quality": "pass",
            "asr_edit_op": "substitute",
            "lexicon_status": "g2p_en",
        }])
        decision = word_mispronunciation_detector(features).iloc[0]["mispronunciation_decision"]
        self.assertEqual(decision, "uncertain_review")


if __name__ == "__main__":
    unittest.main()
