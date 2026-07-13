import unittest

import numpy as np
import pandas as pd

from phoneme_assessment.mdd import assert_speaker_isolation, parse_phone_sequence, source_to_mdd_manifest


class MDDTests(unittest.TestCase):
    def test_manifest_label_mapping(self):
        source = pd.DataFrame(
            [
                {
                    "utterance_id": "u1",
                    "speaker_id": "s1",
                    "audio_path": "a.wav",
                    "target_phone": "AH0",
                    "start_ms": 0,
                    "end_ms": 100,
                    "gold_binary": 1,
                    "split": "train",
                },
                {
                    "utterance_id": "u2",
                    "speaker_id": "s2",
                    "audio_path": "b.wav",
                    "target_phone": "R",
                    "start_ms": 100,
                    "end_ms": 200,
                    "gold_binary": 0,
                    "split": "test",
                },
            ]
        )
        out = source_to_mdd_manifest(source)
        self.assertEqual(out.loc[0, "label"], 0)
        self.assertEqual(out.loc[1, "label"], 1)
        self.assertEqual(out.loc[0, "target_phone"], "AH")

    def test_speaker_isolation(self):
        frame = pd.DataFrame({"speaker_id": ["s1", "s1"], "split": ["train", "test"]})
        with self.assertRaises(ValueError):
            assert_speaker_isolation(frame)

    def test_parse_phone_sequence(self):
        self.assertEqual(parse_phone_sequence("W, IY0 K"), ["W", "IY", "K"])


if __name__ == "__main__":
    unittest.main()
