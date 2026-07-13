from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import wave

import numpy as np
from scipy.signal import resample_poly


TARGET_RATE = 16000


@dataclass
class AudioReport:
    path: str
    exists: bool
    readable: bool
    duration_sec: float | None
    sample_rate: int | None
    channels: int | None
    sample_width_bytes: int | None
    rms: float | None
    needs_preprocess: bool
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def inspect_audio(path: Path, target_rate: int = TARGET_RATE) -> AudioReport:
    warnings: list[str] = []
    if not path.exists():
        return AudioReport(str(path), False, False, None, None, None, None, None, True, ["audio_file_not_found"])
    try:
        rate, channels, width, signal = read_pcm_wav(path)
    except Exception as error:
        return AudioReport(
            str(path),
            True,
            False,
            None,
            None,
            None,
            None,
            None,
            True,
            [f"audio_not_readable:{error}"],
        )
    duration = len(signal) / rate if rate > 0 else 0.0
    rms = float(np.sqrt(np.mean(np.square(signal)))) if len(signal) else 0.0
    if channels != 1:
        warnings.append("audio_not_mono")
    if rate != target_rate:
        warnings.append("audio_sample_rate_not_16k")
    if width != 2:
        warnings.append("audio_not_16bit_pcm")
    if duration <= 0.05:
        warnings.append("audio_duration_too_short")
    if duration > 120.0:
        warnings.append("audio_duration_too_long")
    if rms < 1e-4:
        warnings.append("audio_rms_very_low")
    return AudioReport(
        path=str(path),
        exists=True,
        readable=True,
        duration_sec=round(duration, 6),
        sample_rate=rate,
        channels=channels,
        sample_width_bytes=width,
        rms=round(rms, 8),
        needs_preprocess=bool(warnings),
        warnings=warnings,
    )


def preprocess_audio(
    input_path: Path,
    output_path: Path,
    trim_silence: bool = False,
    silence_threshold: float = 0.01,
    silence_pad_ms: float = 100.0,
    target_rate: int = TARGET_RATE,
) -> AudioReport:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rate, channels, _width, signal = read_pcm_wav(input_path)
        if channels > 1:
            signal = signal.reshape(-1, channels).mean(axis=1)
    except Exception:
        signal, rate = _decode_with_ffmpeg(input_path, target_rate)
    signal = signal.astype(np.float32, copy=False)
    if rate != target_rate:
        divisor = math.gcd(rate, target_rate)
        signal = resample_poly(signal, target_rate // divisor, rate // divisor).astype(np.float32)
        rate = target_rate
    if trim_silence:
        signal = trim_long_edge_silence(signal, rate, silence_threshold, silence_pad_ms)
    write_pcm_wav(output_path, rate, signal)
    return inspect_audio(output_path, target_rate=target_rate)


def read_pcm_wav(path: Path) -> tuple[int, int, int, np.ndarray]:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        raw = handle.readframes(handle.getnframes())
    signal = _pcm_bytes_to_float32(raw, width)
    return rate, channels, width, signal


def _pcm_bytes_to_float32(raw: bytes, width: int) -> np.ndarray:
    if width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (data - 128.0) / 128.0
    if width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if width == 3:
        data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = data[:, 0] | (data[:, 1] << 8) | (data[:, 2] << 16)
        sign_bit = 1 << 23
        values = (values ^ sign_bit) - sign_bit
        return values.astype(np.float32) / float(1 << 23)
    if width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(1 << 31)
    raise ValueError(f"unsupported PCM sample width {width}")


def write_pcm_wav(path: Path, rate: int, signal: np.ndarray) -> None:
    clipped = np.clip(signal, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def trim_long_edge_silence(
    signal: np.ndarray,
    rate: int,
    threshold: float = 0.01,
    pad_ms: float = 100.0,
    frame_ms: float = 20.0,
) -> np.ndarray:
    if len(signal) == 0:
        return signal
    frame = max(1, int(rate * frame_ms / 1000.0))
    active = []
    for start in range(0, len(signal), frame):
        chunk = signal[start:start + frame]
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
        active.append(rms >= threshold)
    if not any(active):
        return signal
    first = active.index(True) * frame
    last = (len(active) - 1 - active[::-1].index(True) + 1) * frame
    pad = int(rate * pad_ms / 1000.0)
    first = max(0, first - pad)
    last = min(len(signal), last + pad)
    return signal[first:last]


def write_report(report: AudioReport, path: Path | None) -> None:
    payload = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    print(payload)


def _decode_with_ffmpeg(input_path: Path, target_rate: int) -> tuple[np.ndarray, int]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("audio is not readable as PCM wav and ffmpeg is not available for conversion")
    with tempfile.TemporaryDirectory() as tmp:
        decoded = Path(tmp) / "decoded.wav"
        command = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            str(target_rate),
            "-sample_fmt",
            "s16",
            str(decoded),
        ]
        subprocess.run(command, check=True)
        rate, _channels, _width, signal = read_pcm_wav(decoded)
    return signal, rate
