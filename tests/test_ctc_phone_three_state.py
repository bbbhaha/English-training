from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pronunciation.ctc_phone_diagnosis import (
    apply_phone_three_state_model,
    arpabet_to_ipa_tokens,
    build_ctc_target_sequence,
    dual_phone_presence_guard,
    force_confirmed_word_deletions,
    summarize_three_state_phones,
)


class CtcPhoneThreeStateTests(unittest.TestCase):
    def test_arpabet_maps_to_l2_arctic_ipa_tokens(self):
        self.assertEqual(arpabet_to_ipa_tokens("TH"), ("θ",))
        self.assertEqual(arpabet_to_ipa_tokens("CH"), ("t", "͡", "ʃ"))
        self.assertEqual(arpabet_to_ipa_tokens("AY1"), ("a", "ɪ"))

    def test_target_sequence_preserves_phone_and_word_spans(self):
        frame = pd.DataFrame(
            [
                {"word": "SHE", "word_index": 0, "target_phone": "SH"},
                {"word": "SHE", "word_index": 0, "target_phone": "IY"},
                {"word": "SEES", "word_index": 1, "target_phone": "S"},
            ]
        )
        vocab = {"ʃ": 1, "i": 2, "s": 3, " ": 4}
        labels, spans, available = build_ctc_target_sequence(frame, vocab)
        self.assertEqual(labels, [1, 2, 4, 3])
        self.assertEqual(spans, [(0, 1), (1, 2), (3, 4)])
        self.assertEqual(available, [True, True, True])

    def test_rule_fallback_emits_exact_three_states(self):
        prediction = pd.DataFrame(
            [
                {"word": "A", "word_index": 0, "phone_index": 0, "target_phone": "AH"},
                {"word": "BAD", "word_index": 1, "phone_index": 1, "target_phone": "B"},
                {"word": "MISS", "word_index": 2, "phone_index": 2, "target_phone": "M"},
            ]
        )
        evidence = pd.DataFrame(
            [
                self._evidence(0, -10.0, -10.0, 8.0),
                self._evidence(1, -10.0, 10.0, -8.0),
                self._evidence(2, 10.0, -10.0, 0.0),
            ]
        )
        result = apply_phone_three_state_model(
            prediction,
            evidence,
            classifier_path=Path("missing-three-state-model.joblib"),
        )
        self.assertEqual(result["phone_state"].tolist(), ["correct", "mispronounced", "deleted"])
        self.assertEqual(result["phone_state_zh"].tolist(), ["读对", "读错", "漏读"])
        self.assertEqual(result["phone_group"].tolist(), ["vowel", "stop", "nasal"])

    def test_confirmed_word_deletion_overrides_phone_state(self):
        phones = pd.DataFrame(
            [
                {
                    "word_index": 1,
                    "phone_state": "correct",
                    "phone_probability_deleted": 0.1,
                    "phone_probability_correct": 0.8,
                    "phone_probability_mispronounced": 0.1,
                }
            ]
        )
        summary = pd.DataFrame(
            [{"word_index": 1, "final_error_type": "deletion", "deletion_decision": "deletion"}]
        )
        result = force_confirmed_word_deletions(phones, summary)
        self.assertEqual(result.loc[0, "phone_state"], "deleted")
        self.assertEqual(result.loc[0, "phone_decision"], "true_error")
        self.assertEqual(result.loc[0, "phone_error_type"], "deletion")

    def test_target_match_suppresses_substitution_false_alarm(self):
        prediction = pd.DataFrame(
            [{"word": "WHAT", "word_index": 0, "phone_index": 0, "target_phone": "T"}]
        )
        evidence = pd.DataFrame([self._evidence(0, -10.0, 10.0, 8.0, recognized="T")])
        result = apply_phone_three_state_model(
            prediction,
            evidence,
            classifier_path=Path("missing-three-state-model.joblib"),
        )
        self.assertEqual(result.loc[0, "phone_state"], "correct")
        self.assertTrue(result.loc[0, "phone_equivalence_guard"])

    def test_narrow_vowel_variant_is_correct_but_real_difference_is_not(self):
        prediction = pd.DataFrame(
            [
                {"word": "WAS", "word_index": 0, "phone_index": 0, "target_phone": "AA"},
                {"word": "VERY", "word_index": 1, "phone_index": 1, "target_phone": "V"},
            ]
        )
        evidence = pd.DataFrame(
            [
                self._evidence(0, -10.0, 10.0, -8.0, recognized="AO"),
                self._evidence(1, -10.0, 10.0, -8.0, recognized="F"),
            ]
        )
        result = apply_phone_three_state_model(
            prediction,
            evidence,
            classifier_path=Path("missing-three-state-model.joblib"),
        )
        self.assertEqual(result["phone_state"].tolist(), ["correct", "mispronounced"])

    def test_of_v_devoicing_is_protected_as_connected_speech(self):
        prediction = pd.DataFrame(
            [{"word": "OF", "word_index": 0, "phone_index": 0, "target_phone": "V"}]
        )
        evidence = pd.DataFrame(
            [self._evidence(0, -5.0, 8.0, -5.0, recognized="F")]
        )
        result = apply_phone_three_state_model(
            prediction,
            evidence,
            classifier_path=Path("missing-three-state-model.joblib"),
        )
        self.assertEqual(result.loc[0, "phone_state"], "correct")

    def test_dual_target_match_rejects_implausible_long_phone_deletion(self):
        frame = pd.DataFrame(
            [
                {
                    "primary_target_match": True,
                    "reference_target_match": True,
                    "ctc_deletion_margin": 2.0,
                    "reference_ctc_deletion_margin": 1.0,
                    "duration_ms": 70.0,
                },
                {
                    "primary_target_match": True,
                    "reference_target_match": True,
                    "ctc_deletion_margin": 2.0,
                    "reference_ctc_deletion_margin": 1.0,
                    "duration_ms": 40.0,
                },
            ]
        )
        self.assertEqual(dual_phone_presence_guard(frame).tolist(), [True, False])

    def test_reference_model_gates_phone_deletion(self):
        prediction = pd.DataFrame(
            [
                {"word": "LIKE", "word_index": 0, "phone_index": 0, "target_phone": "K"},
                {"word": "WAS", "word_index": 1, "phone_index": 1, "target_phone": "W"},
            ]
        )
        primary = pd.DataFrame(
            [
                self._evidence(0, 10.0, -10.0, 8.0, recognized="K"),
                self._evidence(1, 10.0, -10.0, 8.0, recognized="V"),
            ]
        )
        reference = pd.DataFrame(
            [
                self._evidence(0, 1.0, -10.0, 8.0, recognized="K"),
                self._evidence(1, 10.0, -10.0, 8.0, recognized="V"),
            ]
        )
        result = apply_phone_three_state_model(
            prediction,
            primary,
            classifier_path=Path("missing-three-state-model.joblib"),
            reference_evidence=reference,
        )
        self.assertEqual(result["phone_state"].tolist(), ["correct", "deleted"])
        self.assertEqual(result["reference_deletion_supported"].tolist(), [False, True])

    def test_word_summary_counts_three_states(self):
        frame = pd.DataFrame(
            [
                {"word": "BOX", "word_index": 0, "phone_state": "correct"},
                {"word": "BOX", "word_index": 0, "phone_state": "mispronounced"},
                {"word": "BOX", "word_index": 0, "phone_state": "deleted"},
            ]
        )
        summary = summarize_three_state_phones(frame)
        self.assertEqual(summary.loc[0, "num_phone_correct"], 1)
        self.assertEqual(summary.loc[0, "num_phone_mispronounced"], 1)
        self.assertEqual(summary.loc[0, "num_phone_deleted"], 1)

    @staticmethod
    def _evidence(
        phone_index: int,
        deletion: float,
        substitution: float,
        logit: float,
        *,
        recognized: str = "",
    ) -> dict:
        return {
            "phone_index": phone_index,
            "word_index": phone_index,
            "recognized_phone": recognized,
            "ctc_deletion_margin": deletion,
            "ctc_substitution_margin": substitution,
            "ctc_logit_margin": logit,
            "ctc_phone_model_available": True,
            "ctc_phone_error": "",
        }


if __name__ == "__main__":
    unittest.main()
