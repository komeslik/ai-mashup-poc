"""Energy-based vocal activity detection, overlap mute, and instrumental ducking."""

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


@dataclass(frozen=True)
class InstrDuckResult:
    """Instrumental bed after sidechain-style vocal ducking."""

    segment: AudioSegment
    ducked_frames: int


def _frame_rms(samples: np.ndarray, frame_len: int) -> np.ndarray:
    if samples.size == 0:
        return np.zeros(0, dtype=np.float64)
    n_frames = max(1, int(np.ceil(samples.size / frame_len)))
    pad = n_frames * frame_len - samples.size
    if pad:
        samples = np.pad(samples, (0, pad))
    frames = samples.reshape(n_frames, frame_len)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)


def _segment_to_samples(segment: AudioSegment) -> tuple[np.ndarray, int]:
    samples = np.array(segment.get_array_of_samples(), dtype=np.float64)
    channels = segment.channels
    if channels > 1:
        samples = samples.reshape((-1, channels))
    else:
        samples = samples.reshape((-1, 1))
    return samples, channels


def _samples_to_segment(
    samples: np.ndarray,
    reference: AudioSegment,
) -> AudioSegment:
    max_amp = float(1 << (8 * reference.sample_width - 1)) - 1.0
    flat = np.clip(samples.reshape(-1), -max_amp, max_amp).astype(
        {1: np.int8, 2: np.int16, 4: np.int32}.get(reference.sample_width, np.int16)
    )
    return reference._spawn(flat.tobytes())  # noqa: SLF001


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

    max_amp = float(1 << (8 * segment.sample_width - 1))
    samples = samples / max_amp

    frame_len = max(1, int(segment.frame_rate * frame_ms / 1000.0))
    rms = _frame_rms(samples, frame_len)
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-8))
    return rms_db > threshold_db


def _align_length(secondary: AudioSegment, lead: AudioSegment) -> AudioSegment:
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
    return secondary


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
    secondary = _align_length(secondary, lead)

    lead_mask = vocal_activity_mask(lead, frame_ms=frame_ms, threshold_db=threshold_db)
    sec_mask = vocal_activity_mask(secondary, frame_ms=frame_ms, threshold_db=threshold_db)
    n = min(len(lead_mask), len(sec_mask))
    lead_mask = lead_mask[:n]
    sec_mask = sec_mask[:n]
    overlap = lead_mask & sec_mask

    samples, _channels = _segment_to_samples(secondary)
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

    out = _samples_to_segment(samples, secondary)
    return VadOverlapResult(
        segment=out,
        overlap_frames=int(np.sum(overlap)),
        ducked_frames=ducked,
        muted_frames=muted,
    )


def apply_overlap_mute(
    lead: AudioSegment,
    secondary: AudioSegment,
    *,
    frame_ms: int = 30,
    threshold_db: float = -40.0,
) -> VadOverlapResult:
    """Hard-mute secondary wherever both vocals are active (strict anti-collision)."""
    return apply_overlap_duck(
        lead,
        secondary,
        mute_instead=True,
        frame_ms=frame_ms,
        threshold_db=threshold_db,
    )


def duck_instrumental_under_vocals(
    instrumental: AudioSegment,
    vocals: AudioSegment,
    *,
    duck_db: float = -2.5,
    frame_ms: int = 30,
    threshold_db: float = -40.0,
) -> InstrDuckResult:
    """
    Sidechain-style duck: attenuate instrumental frames while vocals are active.
    """
    instrumental = _align_length(instrumental, vocals)
    vocal_mask = vocal_activity_mask(
        vocals, frame_ms=frame_ms, threshold_db=threshold_db
    )
    samples, _ = _segment_to_samples(instrumental)
    frame_len = max(1, int(instrumental.frame_rate * frame_ms / 1000.0))
    gain = float(10 ** (duck_db / 20.0))
    ducked = 0
    n = len(vocal_mask)
    for i in range(n):
        if not vocal_mask[i]:
            continue
        start = i * frame_len
        end = min(samples.shape[0], start + frame_len)
        if start >= samples.shape[0]:
            break
        samples[start:end] *= gain
        ducked += 1
    out = _samples_to_segment(samples, instrumental)
    return InstrDuckResult(segment=out, ducked_frames=ducked)
