import unittest

import numpy as np

from phoneme_assessment.acoustic import GaussianPhoneModel
from phoneme_assessment.alignment import align_feature_sequence


class AlignmentTests(unittest.TestCase):
    def test_two_phone_boundary(self):
        model_a = GaussianPhoneModel(
            mean=np.array([0.0, 0.0]),
            variance=np.array([0.1, 0.1]),
            frames=100,
        )
        model_b = GaussianPhoneModel(
            mean=np.array([5.0, 0.0]),
            variance=np.array([0.1, 0.1]),
            frames=100,
        )
        features = np.vstack(
            [
                np.tile([0.0, 0.0], (5, 1)),
                np.tile([5.0, 0.0], (5, 1)),
            ]
        )
        result = align_feature_sequence(
            features,
            ["A", "B"],
            {"A": model_a, "B": model_b},
            {"A": 50.0, "B": 50.0},
        )
        self.assertEqual(result.boundaries_ms[0], (0.0, 50.0))
        self.assertEqual(result.boundaries_ms[1], (50.0, 115.0))

    def test_missing_model_is_reported(self):
        model = GaussianPhoneModel(
            mean=np.zeros(2),
            variance=np.ones(2),
            frames=10,
        )
        with self.assertRaisesRegex(ValueError, "No acoustic model"):
            align_feature_sequence(
                np.zeros((3, 2)),
                ["A", "B"],
                {"A": model},
                {},
            )


if __name__ == "__main__":
    unittest.main()
