"""Energy-based vocal activity detection and overlap ducking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydub import AudioSegment


@dataclass(frozen=True)
class VadOverlapResult:
    """Result of applying VAD overlap protection to a secondary vocal."""

    segment: AudioSegment
    overlap_frames: int
    ducked_frames: int
    muted_frames: int


def _frame_rms(samples: np.ndarray, frame_len: int) -> np.ndarray:
    if samples.size == 0:
        return np.zeros(0, dtype=np.float64)
    n_frames = max(1, int(np.ceil(samples.size / frame_len)))
    pad = n_frames * frame_len - samples.size
    if pad:
        samples = np.pad(samples, (0, pad))
    frames = samples.reshape(n_frames, frame_len)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)


def vocal_activity_mask(
    segment: AudioSegment,
    *,
    frame_ms: int = 30,
    threshold_db: float = -40.0,
) -> np.ndarray:
    """
    Return a boolean mask (one entry per frame) where True = vocal likely active.

    Uses short-time RMS vs an absolute dBFS-ish threshold derived from peak scale.
    """
    samples = np.array(segment.get_array_of_samples(), dtype=np.float64)
    if segment.channels > 1:
        samples = samples.reshape((-1, segment.channels)).mean(axis=1)

    # Normalize to [-1, 1] based on sample width.
    max_amp = float(1 << (8 * segment.sample_width - 1))
    samples = samples / max_amp

    frame_len = max(1, int(segment.frame_rate * frame_ms / 1000.0))
    rms = _frame_rms(samples, frame_len)
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-8))
    return rms_db > threshold_db


def apply_overlap_duck(
    lead: AudioSegment,
    secondary: AudioSegment,
    *,
    duck_db: float = -18.0,
    mute_instead: bool = False,
    frame_ms: int = 30,
    threshold_db: float = -40.0,
) -> VadOverlapResult:
    """
    When lead and secondary are both active, attenuate or mute secondary frames.

    Keeps lead untouched. Aligns lengths first.
    """
    if len(secondary) != len(lead):
        if len(secondary) > len(lead):
            secondary = secondary[: len(lead)]
        else:
            secondary = secondary + AudioSegment.silent(
                duration=len(lead) - len(secondary),
                frame_rate=secondary.frame_rate,
            )

    if secondary.frame_rate != lead.frame_rate:
        secondary = secondary.set_frame_rate(lead.frame_rate)
    if secondary.channels != lead.channels:
        secondary = secondary.set_channels(lead.channels)

    lead_mask = vocal_activity_mask(lead, frame_ms=frame_ms, threshold_db=threshold_db)
    sec_mask = vocal_activity_mask(secondary, frame_ms=frame_ms, threshold_db=threshold_db)
    n = min(len(lead_mask), len(sec_mask))
    lead_mask = lead_mask[:n]
    sec_mask = sec_mask[:n]
    overlap = lead_mask & sec_mask

    samples = np.array(secondary.get_array_of_samples(), dtype=np.float64)
    channels = secondary.channels
    if channels > 1:
        samples = samples.reshape((-1, channels))
    else:
        samples = samples.reshape((-1, 1))

    frame_len = max(1, int(secondary.frame_rate * frame_ms / 1000.0))
    gain = 0.0 if mute_instead else float(10 ** (duck_db / 20.0))
    ducked = 0
    muted = 0

    for i in range(n):
        if not overlap[i]:
            continue
        start = i * frame_len
        end = min(samples.shape[0], start + frame_len)
        if start >= samples.shape[0]:
            break
        samples[start:end] *= gain
        if mute_instead:
            muted += 1
        else:
            ducked += 1

    flat = samples.reshape(-1)
    # Clip to int range for sample width.
    max_amp = float(1 << (8 * secondary.sample_width - 1)) - 1.0
    flat = np.clip(flat, -max_amp, max_amp).astype(
        {1: np.int8, 2: np.int16, 4: np.int32}.get(secondary.sample_width, np.int16)
    )

    out = secondary._spawn(flat.tobytes())  # noqa: SLF001 — pydub public-ish pattern
    return VadOverlapResult(
        segment=out,
        overlap_frames=int(np.sum(overlap)),
        ducked_frames=ducked,
        muted_frames=muted,
    )
