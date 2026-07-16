from pathlib import Path
import sys
import unittest

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pronunciation.ctc_word_deletion import score_deletion_hypotheses
from pronunciation.word_deletion_model import word_deletion_detector


class CtcWordDeletionTests(unittest.TestCase):
    def test_deleted_hypothesis_wins_when_middle_label_has_no_acoustic_support(self):
        logits = torch.tensor(
            [
                [5.0, 0.0, -8.0, -8.0],
                [0.0, 7.0, -8.0, -8.0],
                [6.0, 0.0, -8.0, 0.0],
                [0.0, -8.0, -8.0, 7.0],
                [6.0, -8.0, -8.0, 0.0],
            ]
        ).log_softmax(dim=-1)
        result = score_deletion_hypotheses(logits, [1, 2, 3], [(1, 2)], blank_id=0)[0]
        self.assertGreater(result["ctc_deletion_margin"], 0.0)
        self.assertGreater(result["ctc_deletion_score"], 0.5)

    def test_asr_alone_is_possible_not_confirmed_deletion(self):
        summary = pd.DataFrame([{
            "word_index": 1,
            "word": "AMERICA",
            "phone_count": 7,
            "word_duration_ms": 500.0,
            "alignment_quality": "pass",
            "lexicon_status": "cmudict",
        }])
        asr = pd.DataFrame([{
            "word_index": 1,
            "asr_missing_word": True,
            "asr_confidence": 0.8,
            "asr_context_support": 1.0,
            "asr_missing_confidence": 0.8,
        }])
        result = word_deletion_detector(summary, asr).iloc[0]
        self.assertEqual(result["deletion_decision"], "possible_deletion")

    def test_asr_and_alignment_free_ctc_confirm_deletion(self):
        summary = pd.DataFrame([{
            "word_index": 1,
            "word": "AMERICA",
            "phone_count": 7,
            "word_duration_ms": 500.0,
            "alignment_quality": "pass",
            "lexicon_status": "cmudict",
        }])
        asr = pd.DataFrame([{
            "word_index": 1,
            "asr_missing_word": True,
            "asr_confidence": 0.8,
            "asr_context_support": 1.0,
            "asr_missing_confidence": 0.8,
        }])
        ctc = pd.DataFrame([{
            "word_index": 1,
            "ctc_deletion_available": True,
            "ctc_deletion_score": 0.9,
        }])
        result = word_deletion_detector(summary, asr, ctc).iloc[0]
        self.assertEqual(result["deletion_decision"], "deletion")

    def test_unavailable_word_is_not_deletion(self):
        summary = pd.DataFrame([{
            "word_index": 0,
            "word": "ZZZXQ",
            "phone_count": 1,
            "word_duration_ms": 0.0,
            "alignment_quality": "bad",
            "lexicon_status": "failed",
        }])
        asr = pd.DataFrame([{"word_index": 0, "asr_missing_word": True}])
        result = word_deletion_detector(summary, asr).iloc[0]
        self.assertEqual(result["deletion_decision"], "not_judged")


if __name__ == "__main__":
    unittest.main()
