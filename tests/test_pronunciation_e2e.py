from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pronunciation.alignment import align_audio_to_text, judge_alignment_quality
from pronunciation.g2p import text_to_phones


class PronunciationE2ETests(unittest.TestCase):
    def test_g2p_word_and_phone_mapping(self):
        result = text_to_phones("WE CALL IT BEAR")
        self.assertEqual(result.phone_sequence, ["W", "IY", "K", "AO", "L", "IH", "T", "B", "EH", "R"])
        self.assertEqual(result.words[0]["word"], "WE")
        self.assertEqual(result.words[0]["phone_index_start"], 0)
        self.assertEqual(result.words[-1]["phone_index_end"], 9)

    def test_oov_is_marked(self):
        result = text_to_phones("ZZZUNKNOWNWORD")
        self.assertEqual(result.words[0]["word"], "ZZZUNKNOWNWORD")
        self.assertIn(result.words[0]["lexicon_status"], {"g2p_en", "phonemizer", "failed"})
        if result.words[0]["lexicon_status"] == "failed":
            self.assertEqual(result.words[0]["phones"], ["<UNK>"])
            self.assertEqual(result.words[0]["g2p_status"], "failed")
            self.assertEqual(result.phones[0]["target_phone"], "<UNK>")

    def test_alignment_quality_bad_for_extreme_duration(self):
        self.assertEqual(
            judge_alignment_quality(
                duration_ms=700.0,
                phone_count=3,
                total_phone_duration_ms=1000.0,
                audio_duration_ms=1000.0,
            ),
            "bad",
        )

    def test_alignment_quality_pass_for_reasonable_duration(self):
        self.assertEqual(
            judge_alignment_quality(
                duration_ms=120.0,
                phone_count=3,
                total_phone_duration_ms=900.0,
                audio_duration_ms=1000.0,
            ),
            "pass",
        )

    def test_alignment_failure_returns_fallback_phone_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "new_audio.wav"
            _write_test_wav(wav_path)
            frame, _g2p = align_audio_to_text(
                wav_path,
                text="WE CALL IT BEAR",
                models_path=Path(tmp) / "missing_models.joblib",
            )
        self.assertFalse(frame.empty)
        self.assertIn("target_phone", frame.columns)
        self.assertIn("start_ms", frame.columns)
        self.assertIn("end_ms", frame.columns)
        self.assertEqual(frame["target_phone"].tolist(), ["W", "IY", "K", "AO", "L", "IH", "T", "B", "EH", "R"])
        self.assertTrue((frame["alignment_quality"] == "bad").all())
        self.assertTrue(frame["review_reason"].str.contains("alignment_failed").all())
        self.assertTrue(frame["review_reason"].str.contains("possible_text_audio_mismatch").all())

    def test_prediction_csv_generated_when_alignment_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wav_path = tmp_path / "MAGA.wav"
            output = tmp_path / "prediction.csv"
            alignment_output = tmp_path / "alignment.csv"
            _write_test_wav(wav_path)
            command = [
                sys.executable,
                str(ROOT / "scripts" / "predict_pronunciation.py"),
                "--audio",
                str(wav_path),
                "--text",
                "MAKE AMERICA GREAT AGAIN",
                "--output",
                str(output),
                "--alignment-output",
                str(alignment_output),
                "--alignment-models",
                str(tmp_path / "missing_models.joblib"),
                "--phase1-model",
                str(tmp_path / "missing_phase1.joblib"),
                "--manual-calibrator",
                str(tmp_path / "missing_calibrator.joblib"),
            ]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
            self.assertIn("Wrote pronunciation prediction", result.stdout)
            self.assertTrue(output.exists())
            rows = _read_csv(output)
        self.assertFalse(rows.empty)
        self.assertIn("target_phone", rows.columns)
        self.assertTrue(rows["target_phone"].fillna("").str.len().gt(0).any())
        self.assertTrue((rows["alignment_quality"] == "bad").all())
        self.assertTrue((rows["decision"] == "uncertain_review").all())
        self.assertTrue((rows["error_type"] == "alignment_issue").all())
        self.assertTrue((rows["confidence"].astype(float) == 0.0).all())
        self.assertTrue((rows["phone_error_probability"].astype(float) >= 0.5).all())
        self.assertTrue((rows["phone_error_percent"].astype(float) >= 50.0).all())
        self.assertTrue((rows["phone_decision"] == "uncertain_review").all())
        self.assertTrue((rows["phone_error_type"] == "alignment_issue").all())
        self.assertTrue(rows["review_reason"].str.contains("Alignment failed", regex=False).all())

def _write_test_wav(path: Path, rate: int = 16000, duration_sec: float = 1.0) -> None:
    samples = np.arange(int(rate * duration_sec), dtype=np.float32)
    signal = 0.1 * np.sin(2 * np.pi * 220.0 * samples / rate)
    pcm = (signal * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def _read_csv(path: Path):
    import pandas as pd

    return pd.read_csv(path)


if __name__ == "__main__":
    unittest.main()
