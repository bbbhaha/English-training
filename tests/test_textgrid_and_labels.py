from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phoneme_assessment.l2_arctic import parse_phone_label
from phoneme_assessment.phones import normalize_phone, phone_group
from phoneme_assessment.speechocean import deterministic_dev_speakers
from phoneme_assessment.textgrid import read_interval_tiers


class ParsingTests(unittest.TestCase):
    def test_error_labels(self):
        self.assertEqual(parse_phone_label("DH, D, s"), ("DH", "D", "substitution"))
        self.assertEqual(parse_phone_label("R, sil, d"), ("R", "sil", "deletion"))
        self.assertEqual(parse_phone_label("sil, T*, a"), ("sil", "T*", "addition"))
        self.assertEqual(parse_phone_label("AE1"), ("AE1", "AE1", "correct"))

    def test_phone_normalization(self):
        self.assertEqual(normalize_phone("ao1*"), "AO")
        self.assertEqual(phone_group("TH"), "fricative")

    def test_minimal_textgrid(self):
        content = '''File type = "ooTextFile"
Object class = "TextGrid"
item [1]:
    class = "IntervalTier"
    name = "phones"
    intervals [1]:
        xmin = 0
        xmax = 0.1
        text = "T"
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.TextGrid"
            path.write_text(content, encoding="utf-8")
            tiers = read_interval_tiers(path)
        self.assertEqual(tiers["phones"][0].text, "T")
        self.assertEqual(tiers["phones"][0].end, 0.1)

    def test_speaker_split_is_deterministic(self):
        speakers = {f"{index:04d}" for index in range(10)}
        first = deterministic_dev_speakers(speakers, 7)
        second = deterministic_dev_speakers(speakers, 7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)


if __name__ == "__main__":
    unittest.main()
