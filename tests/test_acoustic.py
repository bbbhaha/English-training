from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phoneme_assessment.acoustic import fit_phone_models, gop_equivalent_score


class AcousticTests(unittest.TestCase):
    def test_target_likelihood_ratio(self):
        observations = {
            "AA": [np.zeros((50, 3), dtype=np.float32)],
            "T": [np.full((50, 3), 5.0, dtype=np.float32)],
        }
        models = fit_phone_models(observations)
        score, competitor, _, _ = gop_equivalent_score(
            np.zeros((4, 3), dtype=np.float32), "AA", models
        )
        self.assertGreater(score, 0)
        self.assertEqual(competitor, "T")


if __name__ == "__main__":
    unittest.main()
