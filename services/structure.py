"""Musical structure helpers: downbeat-aware phrase/section segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import librosa
import numpy as np

from services.audio import get_beats


SectionLabel = Literal["intro", "low", "high", "outro"]


@dataclass(frozen=True)
class Phrase:
    """A phrase span on a beat/downbeat grid."""

    index: int
    start_beat: int
    end_beat: int
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class Section:
    """A timed musical section with a heuristic energy label."""

    index: int
    start_sec: float
    end_sec: float
    label: SectionLabel
    energy: float
    bars: int

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def to_prompt_dict(self) -> dict:
        return {
            "index": self.index,
            "label": self.label,
            "start_sec": round(self.start_sec, 2),
            "end_sec": round(self.end_sec, 2),
            "energy": round(self.energy, 3),
            "bars": self.bars,
        }


def _estimate_downbeat_indices(beat_times: np.ndarray, y: np.ndarray, sr: int) -> np.ndarray:
    """
    Estimate which beat indices are downbeats (bar starts) in 4/4.

    Combines onset-strength peaks at beats with a Foote-style novelty nudge, then
    picks a phase (0..3) that maximizes energy on every 4th beat.
    """
    n = int(beat_times.size)
    if n < 4:
        return np.arange(0, n, max(n, 1), dtype=int)

    onset = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.frames_to_time(np.arange(len(onset)), sr=sr)
    beat_onset = np.interp(beat_times, onset_times, onset)

    # Novelty from lag-1 difference of beat-sync chroma self-similarity proxy.
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    beat_frames = librosa.time_to_frames(beat_times, sr=sr)
    beat_frames = np.unique(np.clip(beat_frames, 0, chroma.shape[1] - 1))
    if beat_frames.size >= 2:
        sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
        # Local novelty: 1 - corr with previous beat chroma.
        novelty = np.zeros(sync.shape[1], dtype=np.float64)
        for i in range(1, sync.shape[1]):
            a = sync[:, i - 1]
            b = sync[:, i]
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
            novelty[i] = 1.0 - float(np.dot(a, b) / denom)
        # Align novelty length to beats.
        if novelty.size < n:
            novelty = np.pad(novelty, (0, n - novelty.size))
        novelty = novelty[:n]
    else:
        novelty = np.zeros(n, dtype=np.float64)

    score = 0.7 * (beat_onset / (np.max(beat_onset) + 1e-8)) + 0.3 * novelty
    best_phase = 0
    best_val = -np.inf
    for phase in range(4):
        val = float(np.sum(score[phase::4]))
        if val > best_val:
            best_val = val
            best_phase = phase
    return np.arange(best_phase, n, 4, dtype=int)


def segment_phrases(
    file_path: str,
    *,
    beats_per_phrase: int = 8,
    bars_per_phrase: int | None = None,
    max_phrases: int | None = 12,
    use_downbeats: bool = True,
) -> list[Phrase]:
    """
    Split audio into phrase chunks on a downbeat-aware beat grid.

    Default ``beats_per_phrase=8`` ≈ two bars in 4/4. When ``bars_per_phrase`` is
    set (e.g. 8 or 16), phrase length is ``bars_per_phrase * 4`` beats.
    """
    if bars_per_phrase is not None:
        beats_per_phrase = int(bars_per_phrase) * 4
    if beats_per_phrase < 2:
        raise ValueError("beats_per_phrase must be >= 2")

    y, sr = librosa.load(file_path, sr=None, mono=True)
    _, beat_times = get_beats(file_path)
    duration = float(librosa.get_duration(y=y, sr=sr))
    n_beats = int(beat_times.size)

    if n_beats < beats_per_phrase + 1:
        return [
            Phrase(
                index=0,
                start_beat=0,
                end_beat=max(n_beats - 1, 1),
                start_sec=0.0,
                end_sec=duration,
            )
        ]

    if use_downbeats:
        downbeats = _estimate_downbeat_indices(beat_times, y, sr)
        if downbeats.size == 0:
            starts = list(range(0, n_beats - beats_per_phrase, beats_per_phrase))
        else:
            # Prefer phrase starts on downbeats, stepping by phrase length in beats.
            starts = []
            for db in downbeats.tolist():
                if db + beats_per_phrase < n_beats:
                    if not starts or db >= starts[-1] + beats_per_phrase:
                        starts.append(int(db))
            if not starts:
                starts = list(range(0, n_beats - beats_per_phrase, beats_per_phrase))
    else:
        starts = list(range(0, n_beats - beats_per_phrase, beats_per_phrase))

    phrases: list[Phrase] = []
    for index, start_beat in enumerate(starts):
        end_beat = start_beat + beats_per_phrase
        if end_beat >= n_beats:
            break
        start_sec = float(beat_times[start_beat])
        end_sec = float(beat_times[end_beat])
        if end_sec <= start_sec + 0.25:
            continue
        phrases.append(
            Phrase(
                index=index,
                start_beat=start_beat,
                end_beat=end_beat,
                start_sec=start_sec,
                end_sec=end_sec,
            )
        )
        if max_phrases is not None and len(phrases) >= max_phrases:
            break

    if not phrases:
        phrases.append(
            Phrase(
                index=0,
                start_beat=0,
                end_beat=min(beats_per_phrase, n_beats - 1),
                start_sec=0.0,
                end_sec=min(duration, float(beat_times[min(beats_per_phrase, n_beats - 1)])),
            )
        )

    return phrases


def onset_strength_peak_count(file_path: str) -> int:
    """Rough activity score used when choosing a denser instrumental bed."""
    y, sr = librosa.load(file_path, sr=None, mono=True)
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    threshold = float(np.median(onset) + np.std(onset))
    return int(np.sum(onset > threshold))


def _phrase_rms_energy(
    y: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
) -> float:
    start = max(0, int(start_sec * sr))
    end = min(len(y), int(end_sec * sr))
    if end <= start:
        return 0.0
    chunk = y[start:end]
    return float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))


def _label_sections(
    energies: list[float],
    *,
    high_quantile: float = 0.6,
) -> list[SectionLabel]:
    """Heuristic intro/low/high/outro labels from relative RMS energy."""
    n = len(energies)
    if n == 0:
        return []
    if n == 1:
        return ["high"]

    arr = np.asarray(energies, dtype=np.float64)
    threshold = float(np.quantile(arr, high_quantile))
    labels: list[SectionLabel] = []
    for i, energy in enumerate(energies):
        if i == 0 and energy < threshold:
            labels.append("intro")
        elif i == n - 1 and energy < threshold:
            labels.append("outro")
        elif energy >= threshold:
            labels.append("high")
        else:
            labels.append("low")
    # Ensure at least one high section when possible.
    if "high" not in labels and n >= 2:
        labels[int(np.argmax(arr))] = "high"
    return labels


def detect_sections(
    file_path: str,
    *,
    bars_per_section: int = 8,
    max_sections: int | None = 8,
) -> list[Section]:
    """
    Detect timed sections on a downbeat-aware bar grid.

    Labels are energy heuristics (intro/low/high/outro), not lyric tags.
    Default section length is 8 bars ≈ a chorus-sized chunk in 4/4.
    """
    phrases = segment_phrases(
        file_path,
        bars_per_phrase=bars_per_section,
        max_phrases=max_sections,
        use_downbeats=True,
    )
    y, sr = librosa.load(file_path, sr=None, mono=True)
    energies = [
        _phrase_rms_energy(y, sr, phrase.start_sec, phrase.end_sec) for phrase in phrases
    ]
    labels = _label_sections(energies)
    sections: list[Section] = []
    for phrase, energy, label in zip(phrases, energies, labels):
        sections.append(
            Section(
                index=phrase.index,
                start_sec=phrase.start_sec,
                end_sec=phrase.end_sec,
                label=label,
                energy=float(energy),
                bars=bars_per_section,
            )
        )
    return sections


def pick_section(sections: list[Section], index: int) -> Section:
    """Clamp *index* into range; empty list raises."""
    if not sections:
        raise ValueError("No sections available")
    if index < 0:
        index = 0
    if index >= len(sections):
        index = len(sections) - 1
    return sections[index]