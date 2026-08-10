"""Mashability scoring: harmonic, 12-bin sub-beat rhythmic, and spectral."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from services.audio import (
    DEFAULT_MAX_KEY_SHIFT,
    DEFAULT_WINDOW_BEATS,
    HarmonicAlignment,
    _beat_sync_chroma,
    _normalize_chroma_columns,
    _window_correlation,
    get_beats,
    get_key,
    semitones_to_match_key,
)

# Paper-style sub-beat resolution (straight vs swing discrimination).
SUB_BEATS = 12
# Below this rhythmic similarity, prefer alternate over harmony.
LOW_RHYTHM_THRESHOLD = 0.35


@dataclass(frozen=True)
class MashabilityWeights:
    harmonic: float = 0.6
    rhythmic: float = 0.25
    spectral: float = 0.15

    def normalized(self) -> MashabilityWeights:
        total = self.harmonic + self.rhythmic + self.spectral
        if total <= 0:
            return MashabilityWeights()
        return MashabilityWeights(
            harmonic=self.harmonic / total,
            rhythmic=self.rhythmic / total,
            spectral=self.spectral / total,
        )


@dataclass(frozen=True)
class MashabilityAlignment(HarmonicAlignment):
    """HarmonicAlignment plus factor scores for debugging / policy."""

    harmonic_score: float = 0.0
    rhythmic_score: float = 0.0
    spectral_score: float = 0.0


def _beat_sync_bands(y: np.ndarray, sr: int, beat_times: np.ndarray) -> np.ndarray:
    """Return ``(3, n_beats)`` low/mid/high band energies (paper-ish cutoffs)."""
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    low_m = (freqs >= 20) & (freqs < 220)
    mid_m = (freqs >= 220) & (freqs < 1760)
    high_m = freqs >= 1760
    low = S[low_m].mean(axis=0) if np.any(low_m) else S.mean(axis=0)
    mid = S[mid_m].mean(axis=0) if np.any(mid_m) else S.mean(axis=0)
    high = S[high_m].mean(axis=0) if np.any(high_m) else S.mean(axis=0)
    bands = np.vstack([low, mid, high])
    beat_frames = librosa.time_to_frames(beat_times, sr=sr)
    beat_frames = np.unique(np.clip(beat_frames, 0, bands.shape[1] - 1))
    if beat_frames.size < 2:
        return np.mean(bands, axis=1, keepdims=True).astype(np.float64)
    synced = librosa.util.sync(bands, beat_frames, aggregate=np.mean)
    return np.asarray(synced, dtype=np.float64)


def _subbeat_rhythm_matrix(
    y: np.ndarray,
    sr: int,
    beat_times: np.ndarray,
    *,
    sub_beats: int = SUB_BEATS,
) -> np.ndarray:
    """
    Return ``(sub_beats, n_beats-1)`` onset histograms quantized within each beat.

    AutoMashUpper-style 12ths-of-a-beat rhythmic descriptor.
    """
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.frames_to_time(np.arange(len(onset)), sr=sr)
    n_intervals = max(0, len(beat_times) - 1)
    matrix = np.zeros((sub_beats, n_intervals), dtype=np.float64)
    if n_intervals == 0:
        return matrix

    for i in range(n_intervals):
        t0 = float(beat_times[i])
        t1 = float(beat_times[i + 1])
        if t1 <= t0:
            continue
        mask = (onset_times >= t0) & (onset_times < t1)
        if not np.any(mask):
            continue
        rel = (onset_times[mask] - t0) / (t1 - t0)
        bins = np.clip((rel * sub_beats).astype(int), 0, sub_beats - 1)
        weights = onset[mask]
        for b, w in zip(bins, weights):
            matrix[b, i] += float(w)
        col_sum = matrix[:, i].sum()
        if col_sum > 1e-8:
            matrix[:, i] /= col_sum
    return matrix


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    denom = float(np.linalg.norm(flat_a) * np.linalg.norm(flat_b))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(flat_a, flat_b) / denom)


def _spectral_complement(a: np.ndarray, b: np.ndarray) -> float:
    mean_a = np.mean(a, axis=1)
    mean_b = np.mean(b, axis=1)
    sim = _cosine(mean_a, mean_b)
    return float(max(0.0, 1.0 - sim))


def find_best_mashability_alignment(
    vocal_path: str,
    instrumental_path: str,
    *,
    weights: MashabilityWeights | None = None,
    window_beats: int = DEFAULT_WINDOW_BEATS,
    max_key_shift: int = DEFAULT_MAX_KEY_SHIFT,
    fixed_n_steps: int | None = None,
) -> MashabilityAlignment:
    """
    Search beat offset + key rotation maximizing weighted mashability.

    When *fixed_n_steps* is set, lock pitch rotation to that value (global key
    meeting) and only search beat windows / rhythm / spectral fit.

    Rhythm uses 12 sub-beat onset histograms (straight vs swing sensitive).
    """
    w = (weights or MashabilityWeights()).normalized()

    y_v, sr_v = librosa.load(vocal_path, sr=None, mono=True)
    y_i, sr_i = librosa.load(instrumental_path, sr=None, mono=True)
    _, beats_v = get_beats(vocal_path)
    _, beats_i = get_beats(instrumental_path)

    chroma_v = _normalize_chroma_columns(_beat_sync_chroma(y_v, sr_v, beats_v))
    chroma_i = _normalize_chroma_columns(_beat_sync_chroma(y_i, sr_i, beats_i))
    rhythm_v = _subbeat_rhythm_matrix(y_v, sr_v, beats_v)
    rhythm_i = _subbeat_rhythm_matrix(y_i, sr_i, beats_i)
    bands_v = _beat_sync_bands(y_v, sr_v, beats_v)
    bands_i = _beat_sync_bands(y_i, sr_i, beats_i)

    # Rhythm matrix has n_beats-1 columns; chroma/bands have n_beats.
    n_v = min(chroma_v.shape[1], bands_v.shape[1], rhythm_v.shape[1] + 1)
    n_i = min(chroma_i.shape[1], bands_i.shape[1], rhythm_i.shape[1] + 1)
    chroma_v = chroma_v[:, :n_v]
    chroma_i = chroma_i[:, :n_i]
    bands_v = bands_v[:, :n_v]
    bands_i = bands_i[:, :n_i]
    rhythm_v = rhythm_v[:, : max(0, n_v - 1)]
    rhythm_i = rhythm_i[:, : max(0, n_i - 1)]

    win = min(window_beats, n_v, n_i)
    duration_v = float(librosa.get_duration(y=y_v, sr=sr_v))
    duration_i = float(librosa.get_duration(y=y_i, sr=sr_i))

    if win < 2:
        if fixed_n_steps is not None:
            n_steps = int(fixed_n_steps)
        else:
            try:
                n_steps = semitones_to_match_key(get_key(vocal_path), get_key(instrumental_path))
            except ValueError:
                n_steps = 0
        end = min(duration_v, duration_i)
        return MashabilityAlignment(
            n_steps=n_steps,
            score=0.0,
            window_beats=max(win, 1),
            vocal_beat_start=0,
            instrumental_beat_start=0,
            vocal_start_sec=0.0,
            vocal_end_sec=end,
            instrumental_start_sec=0.0,
            instrumental_end_sec=end,
        )

    if fixed_n_steps is not None:
        key_shifts = [int(fixed_n_steps)]
    else:
        key_shifts = list(range(-max_key_shift, max_key_shift + 1))
    best_score = -np.inf
    best: tuple[int, int, int, float, float, float] = (
        key_shifts[0],
        0,
        0,
        0.0,
        0.0,
        0.0,
    )

    vocal_step = 1 if n_v - win < 48 else max(1, (n_v - win) // 24)
    instr_step = 1 if n_i - win < 64 else max(1, (n_i - win) // 32)
    rhythm_win = max(1, win - 1)

    for n_steps in key_shifts:
        rotated = np.roll(chroma_v, n_steps, axis=0)
        for v0 in range(0, n_v - win + 1, vocal_step):
            patch_v = rotated[:, v0 : v0 + win]
            band_patch_v = bands_v[:, v0 : v0 + win]
            r_v0 = min(v0, max(0, rhythm_v.shape[1] - rhythm_win))
            rhythm_patch_v = rhythm_v[:, r_v0 : r_v0 + rhythm_win]
            for i0 in range(0, n_i - win + 1, instr_step):
                patch_i = chroma_i[:, i0 : i0 + win]
                harm = _window_correlation(patch_v, patch_i)
                r_i0 = min(i0, max(0, rhythm_i.shape[1] - rhythm_win))
                rhythm = _cosine(rhythm_patch_v, rhythm_i[:, r_i0 : r_i0 + rhythm_win])
                spectral = _spectral_complement(band_patch_v, bands_i[:, i0 : i0 + win])
                score = w.harmonic * harm + w.rhythmic * rhythm + w.spectral * spectral
                if score > best_score:
                    best_score = score
                    best = (n_steps, v0, i0, harm, rhythm, spectral)

    n_steps, v0, i0, harm_s, rhythm_s, spectral_s = best
    v1 = v0 + win
    i1 = i0 + win

    def _beat_span(beats: np.ndarray, start: int, end: int, duration: float) -> tuple[float, float]:
        t0 = float(beats[start]) if start < len(beats) else 0.0
        if end < len(beats):
            t1 = float(beats[end])
        else:
            t1 = duration
        if t1 <= t0:
            t1 = min(duration, t0 + (60.0 / 120.0) * max(end - start, 1))
        return t0, t1

    v_start, v_end = _beat_span(beats_v, v0, v1, duration_v)
    i_start, i_end = _beat_span(beats_i, i0, i1, duration_i)

    return MashabilityAlignment(
        n_steps=int(n_steps),
        score=float(best_score),
        window_beats=int(win),
        vocal_beat_start=int(v0),
        instrumental_beat_start=int(i0),
        vocal_start_sec=v_start,
        vocal_end_sec=v_end,
        instrumental_start_sec=i_start,
        instrumental_end_sec=i_end,
        harmonic_score=float(harm_s),
        rhythmic_score=float(rhythm_s),
        spectral_score=float(spectral_s),
    )
