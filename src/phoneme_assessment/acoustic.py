"""Dependency-light acoustic features and diagonal Gaussian phone models."""

from __future__ import annotations

from dataclasses import dataclass
import math
import wave
from pathlib import Path

import numpy as np
from scipy.fft import rfft
from scipy.signal import resample_poly


def read_wav_mono(path: Path, target_rate: int = 16000) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        raw = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"Only 16-bit PCM is supported: {path}")
    signal = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        signal = signal.reshape(-1, channels).mean(axis=1)
    if rate != target_rate:
        divisor = math.gcd(rate, target_rate)
        signal = resample_poly(signal, target_rate // divisor, rate // divisor)
        rate = target_rate
    return rate, signal.astype(np.float32, copy=False)


def _hz_to_mel(value: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(value) / 700.0)


def _mel_to_hz(value: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(value) / 2595.0) - 1.0)


def mel_filterbank(rate: int, n_fft: int, n_mels: int = 24) -> np.ndarray:
    mel_points = np.linspace(_hz_to_mel(50), _hz_to_mel(rate / 2), n_mels + 2)
    bins = np.floor((n_fft + 1) * _mel_to_hz(mel_points) / rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = bins[index:index + 3]
        if center > left:
            filters[index, left:center] = (
                np.arange(left, center) - left
            ) / (center - left)
        if right > center:
            filters[index, center:right] = (
                right - np.arange(center, right)
            ) / (right - center)
    return filters


def segment_features(
    signal: np.ndarray,
    rate: int,
    start_ms: float,
    end_ms: float,
    n_mels: int = 24,
) -> np.ndarray:
    start = max(0, round(start_ms * rate / 1000))
    end = min(len(signal), round(end_ms * rate / 1000))
    segment = signal[start:end]
    frame_length = round(0.025 * rate)
    frame_step = round(0.010 * rate)
    if len(segment) < frame_length:
        segment = np.pad(segment, (0, frame_length - len(segment)))
    frame_count = 1 + max(0, (len(segment) - frame_length) // frame_step)
    starts = np.arange(frame_count) * frame_step
    frames = np.stack([segment[s:s + frame_length] for s in starts])
    frames = frames - frames.mean(axis=1, keepdims=True)
    frames *= np.hamming(frame_length)
    n_fft = 512
    power = np.abs(rfft(frames, n=n_fft, axis=1)) ** 2
    log_mel = np.log(np.maximum(power @ mel_filterbank(rate, n_fft, n_mels).T, 1e-10))
    energy = np.log(np.maximum(np.mean(frames ** 2, axis=1, keepdims=True), 1e-10))
    return np.concatenate([log_mel, energy], axis=1).astype(np.float32)


@dataclass
class GaussianPhoneModel:
    mean: np.ndarray
    variance: np.ndarray
    frames: int

    def log_likelihood(self, features: np.ndarray) -> float:
        return float(self.frame_log_likelihood(features).mean())

    def frame_log_likelihood(self, features: np.ndarray) -> np.ndarray:
        """Return one diagonal-Gaussian log likelihood per acoustic frame."""
        value = -0.5 * (
            np.log(2 * np.pi * self.variance)
            + ((features - self.mean) ** 2) / self.variance
        )
        return value.sum(axis=1)


def fit_phone_models(
    observations: dict[str, list[np.ndarray]],
    minimum_frames: int = 40,
) -> dict[str, GaussianPhoneModel]:
    models: dict[str, GaussianPhoneModel] = {}
    for phone, chunks in observations.items():
        matrix = np.concatenate(chunks, axis=0)
        if len(matrix) < minimum_frames:
            continue
        models[phone] = GaussianPhoneModel(
            mean=matrix.mean(axis=0),
            variance=np.maximum(matrix.var(axis=0), 0.25),
            frames=len(matrix),
        )
    return models


def gop_equivalent_score(
    features: np.ndarray,
    target_phone: str,
    models: dict[str, GaussianPhoneModel],
) -> tuple[float, str, float, float]:
    scores = {phone: model.log_likelihood(features) for phone, model in models.items()}
    target_score = scores[target_phone]
    competitor, competitor_score = max(
        ((phone, score) for phone, score in scores.items() if phone != target_phone),
        key=lambda item: item[1],
    )
    return target_score - competitor_score, competitor, target_score, competitor_score
