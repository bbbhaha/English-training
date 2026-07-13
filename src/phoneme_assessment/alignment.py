"""Dependency-light monotonic phone forced alignment."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .acoustic import GaussianPhoneModel, segment_features


@dataclass
class AlignmentResult:
    boundaries_ms: list[tuple[float, float]]
    score_per_frame: float
    active_start_ms: float
    active_end_ms: float
    duration_scale: float


def active_frame_indices(
    features: np.ndarray,
    minimum_internal_silence_frames: int = 6,
) -> np.ndarray:
    """Select speech frames while removing sustained leading/internal silence."""
    if len(features) == 0:
        return np.asarray([], dtype=np.int32)
    energy = features[:, -1]
    floor = float(np.quantile(energy, 0.10))
    peak = float(np.max(energy))
    threshold = max(floor + math.log(3.0), peak - 8.0)
    speech = energy >= threshold
    active = np.flatnonzero(speech)
    if len(active) == 0:
        return np.arange(len(features), dtype=np.int32)

    keep = np.zeros(len(features), dtype=bool)
    keep[max(0, int(active[0]) - 2):min(len(features), int(active[-1]) + 3)] = True

    # Preserve brief low-energy closures and fricative dips, but remove sustained
    # pauses. Two context frames remain on each side to avoid clipping phones.
    quiet = ~speech
    run_start = None
    for index in range(len(features) + 1):
        is_quiet = index < len(features) and bool(quiet[index])
        if is_quiet and run_start is None:
            run_start = index
        elif not is_quiet and run_start is not None:
            if index - run_start >= minimum_internal_silence_frames:
                keep[run_start + 2:max(run_start + 2, index - 2)] = False
            run_start = None
    return np.flatnonzero(keep)


def _duration_ranges(
    expected_frames: np.ndarray,
    total_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.maximum(1, np.floor(expected_frames * 0.30).astype(int))
    maximum = np.maximum(
        minimum,
        np.ceil(expected_frames * 3.0 + 3).astype(int),
    )
    maximum = np.minimum(maximum, total_frames)

    # Guarantee at least one feasible path even for unusually short/long audio.
    while int(minimum.sum()) > total_frames:
        index = int(np.argmax(minimum))
        if minimum[index] <= 1:
            break
        minimum[index] -= 1
    while int(maximum.sum()) < total_frames:
        maximum[int(np.argmin(maximum))] += 1
    return minimum, maximum


def align_feature_sequence(
    features: np.ndarray,
    phones: list[str],
    models: dict[str, GaussianPhoneModel],
    duration_priors_ms: dict[str, float],
    frame_step_ms: float = 10.0,
    frame_length_ms: float = 25.0,
    internal_silence_penalty: float = 18.0,
) -> AlignmentResult:
    """Align a known phone sequence to acoustic frames with segmental Viterbi."""
    if not phones:
        raise ValueError("Cannot align an empty phone sequence")
    missing = sorted(set(phones) - set(models))
    if missing:
        raise ValueError(f"No acoustic model for phones: {', '.join(missing)}")

    frame_indices = active_frame_indices(features)
    active = features[frame_indices]
    if len(active) < len(phones):
        frame_indices = np.arange(len(features), dtype=np.int32)
        active = features
    if len(active) < len(phones):
        raise ValueError(
            f"Only {len(active)} acoustic frames for {len(phones)} phones"
        )

    raw_expected = np.asarray(
        [max(20.0, duration_priors_ms.get(phone, 100.0)) / frame_step_ms
         for phone in phones],
        dtype=np.float64,
    )
    duration_scale = len(active) / float(raw_expected.sum())
    expected = np.maximum(1.0, raw_expected * duration_scale)
    minimum, maximum = _duration_ranges(expected, len(active))

    emissions = np.stack(
        [models[phone].frame_log_likelihood(active) for phone in phones]
    )
    cumulative = np.pad(
        np.cumsum(emissions, axis=1),
        ((0, 0), (1, 0)),
        constant_values=0.0,
    )
    phone_count = len(phones)
    frame_count = len(active)
    negative_infinity = -np.inf
    dp = np.full((phone_count + 1, frame_count + 1), negative_infinity)
    back = np.full((phone_count + 1, frame_count + 1), -1, dtype=np.int32)
    dp[0, 0] = 0.0

    for phone_index in range(1, phone_count + 1):
        remaining_minimum = int(minimum[phone_index:].sum())
        previous_minimum = int(minimum[:phone_index - 1].sum())
        earliest_end = previous_minimum + int(minimum[phone_index - 1])
        latest_end = frame_count - remaining_minimum
        for end in range(earliest_end, latest_end + 1):
            low = max(int(minimum[phone_index - 1]), end - frame_count)
            high = min(int(maximum[phone_index - 1]), end)
            durations = np.arange(low, high + 1)
            starts = end - durations
            prior_scores = dp[phone_index - 1, starts]
            valid = np.isfinite(prior_scores)
            if not np.any(valid):
                continue
            durations = durations[valid]
            starts = starts[valid]
            acoustic = (
                cumulative[phone_index - 1, end]
                - cumulative[phone_index - 1, starts]
            )
            # A phone segment should not span a removed pause. The penalty makes
            # state boundaries gravitate toward sustained low-energy gaps.
            original_span = frame_indices[end - 1] - frame_indices[starts] + 1
            omitted_frames = original_span - durations
            silence_penalty = -internal_silence_penalty * omitted_frames
            sigma = max(1.5, expected[phone_index - 1] * 0.65)
            duration_penalty = -0.5 * (
                (durations - expected[phone_index - 1]) / sigma
            ) ** 2
            candidates = (
                dp[phone_index - 1, starts]
                + acoustic
                + duration_penalty
                + silence_penalty
            )
            best = int(np.argmax(candidates))
            dp[phone_index, end] = candidates[best]
            back[phone_index, end] = int(starts[best])

    if not np.isfinite(dp[phone_count, frame_count]):
        raise ValueError("No feasible alignment path")

    frame_boundaries: list[tuple[int, int]] = []
    end = frame_count
    for phone_index in range(phone_count, 0, -1):
        start = int(back[phone_index, end])
        frame_boundaries.append((start, end))
        end = start
    frame_boundaries.reverse()

    boundaries_ms = []
    for index, (start, end) in enumerate(frame_boundaries):
        start_ms = float(frame_indices[start]) * frame_step_ms
        end_ms = float(frame_indices[end - 1] + 1) * frame_step_ms
        if index == len(frame_boundaries) - 1:
            end_ms += frame_length_ms - frame_step_ms
        boundaries_ms.append((start_ms, end_ms))

    active_start_ms = float(frame_indices[0]) * frame_step_ms
    active_end_ms = (
        float(frame_indices[-1] + 1) * frame_step_ms
        + frame_length_ms
        - frame_step_ms
    )
    return AlignmentResult(
        boundaries_ms=boundaries_ms,
        score_per_frame=float(dp[phone_count, frame_count] / frame_count),
        active_start_ms=active_start_ms,
        active_end_ms=active_end_ms,
        duration_scale=float(duration_scale),
    )


def align_signal(
    signal: np.ndarray,
    rate: int,
    phones: list[str],
    models: dict[str, GaussianPhoneModel],
    duration_priors_ms: dict[str, float],
) -> AlignmentResult:
    duration_ms = len(signal) * 1000.0 / rate
    features = segment_features(signal, rate, 0.0, duration_ms)
    return align_feature_sequence(features, phones, models, duration_priors_ms)
