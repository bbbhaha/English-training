import unittest

import pandas as pd

from webapp.app import apply_phone_diagnosis_display


class WebappPhoneThreeStateDisplayTests(unittest.TestCase):
    def test_frontend_displays_exact_three_states(self):
        prediction = pd.DataFrame(
            [
                self._row(0, "correct", "", 0.05),
                self._row(1, "mispronounced", "mispronunciation", 0.88),
                self._row(2, "deleted", "deletion", 0.97),
            ]
        )
        display = apply_phone_diagnosis_display(prediction)
        self.assertEqual(display["display_decision"].tolist(), ["读对", "读错", "漏读"])
        self.assertEqual(display["display_error_type"].tolist(), ["", "mispronunciation", "deletion"])
        self.assertEqual(display["display_error"].tolist(), ["读对 95%", "读错 88%", "漏读 97%"])

    @staticmethod
    def _row(index: int, state: str, error_type: str, probability: float) -> dict:
        return {
            "word": f"W{index}",
            "word_index": index,
            "phone_index": index,
            "target_phone": "T",
            "alignment_quality": "pass",
            "phone_state": state,
            "phone_decision": "correct" if state == "correct" else "true_error",
            "phone_error_type": error_type,
            "phone_error_probability": probability,
            "phone_error_percent": probability * 100,
            "phone_probability_correct": 1.0 - probability if state == "correct" else 0.0,
            "phone_probability_mispronounced": probability if state == "mispronounced" else 0.0,
            "phone_probability_deleted": probability if state == "deleted" else 0.0,
        }


if __name__ == "__main__":
    unittest.main()
