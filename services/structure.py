"""Musical structure helpers: downbeat-aware phrase/section segmentation + role maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import librosa
import numpy as np
from pydub import AudioSegment

from services.audio import get_beats
from services.vad import vocal_activity_mask


SectionLabel = Literal[
    "intro",
    "verse",
    "buildup",
    "chorus",
    "drop",
    "outro",
    "bridge",
    "other",
    "low",
    "high",
]


@dataclass(frozen=True)
class Phrase:
    """A phrase span on a downbeat-aware beat grid."""

    index: int
    start_beat: int
    end_beat: int
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class Section:
    """A timed musical section with multi-feature role labels."""

    index: int
    start_sec: float
    end_sec: float
    label: SectionLabel
    energy: float
    bars: int
    spectral_centroid: float = 0.0
    vocal_density: float = 0.0
    bar_start: int = 0
    bar_end: int = 8
    name: str = ""
    description: str = ""
    approx_beats: int = 0

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def to_prompt_dict(self) -> dict:
        return {
            "index": self.index,
            "label": self.label,
            "name": self.name or self.label,
            "start_sec": round(self.start_sec, 2),
            "end_sec": round(self.end_sec, 2),
            "energy": round(self.energy, 3),
            "spectral_centroid": round(self.spectral_centroid, 1),
            "vocal_density": round(self.vocal_density, 3),
            "bars": self.bars,
            "bar_start": self.bar_start,
            "bar_end": self.bar_end,
            "approx_beats": self.approx_beats,
            "description": self.description,
        }


def normalize_section_label(raw: str) -> SectionLabel:
    """Map free-form section names to our label enum."""
    s = (raw or "").strip().lower()
    if not s:
        return "other"
    if "intro" in s:
        return "intro"
    if "outro" in s or "fade" in s:
        return "outro"
    if "pre-chorus" in s or "prechorus" in s or "build" in s:
        return "buildup"
    if "drop" in s:
        return "drop"
    if "chorus" in s or "hook" in s or "refrain" in s:
        return "chorus"
    if "bridge" in s or "interlude" in s or "solo" in s or "break" in s:
        return "bridge"
    if "verse" in s or "groove" in s:
        return "verse"
    return "other"


def snap_form_timestamps(
    specs: list[dict],
    measured_duration_sec: float,
    *,
    expected_duration_sec: float | None = None,
) -> list[dict]:
    """
    Scale/clamp LLM section timestamps onto the uploaded file's measured duration.

    Ensures contiguous coverage from 0 → measured_duration.
    """
    if measured_duration_sec <= 0:
        raise ValueError("measured_duration_sec must be positive")
    if not specs:
        return [
            {
                "name": "Full track",
                "label": "verse",
                "start_sec": 0.0,
                "end_sec": measured_duration_sec,
                "approx_beats": 0,
                "description": "",
            }
        ]

    expected = float(expected_duration_sec or 0.0)
    scale = 1.0
    if expected > 1.0 and abs(expected - measured_duration_sec) > 1.5:
        scale = measured_duration_sec / expected

    scaled: list[dict] = []
    for spec in specs:
        start = float(spec.get("start_sec", 0.0)) * scale
        end = float(spec.get("end_sec", start)) * scale
        start = max(0.0, min(measured_duration_sec, start))
        end = max(0.0, min(measured_duration_sec, end))
        if end <= start + 0.05:
            continue
        scaled.append(
            {
                "name": str(spec.get("name") or spec.get("label") or "Section"),
                "label": normalize_section_label(
                    str(spec.get("label") or spec.get("name") or "")
                ),
                "start_sec": start,
                "end_sec": end,
                "approx_beats": int(spec.get("approx_beats") or 0),
                "description": str(spec.get("description") or ""),
            }
        )

    if not scaled:
        return snap_form_timestamps([], measured_duration_sec)

    # Sort and make contiguous.
    scaled.sort(key=lambda s: s["start_sec"])
    contiguous: list[dict] = []
    cursor = 0.0
    for i, sec in enumerate(scaled):
        start = sec["start_sec"] if i == 0 else max(sec["start_sec"], cursor)
        end = sec["end_sec"]
        if i == 0:
            start = 0.0
        if i == len(scaled) - 1:
            end = measured_duration_sec
        if end <= start + 0.05:
            continue
        contiguous.append({**sec, "start_sec": start, "end_sec": end})
        cursor = end

    if not contiguous:
        return snap_form_timestamps([], measured_duration_sec)
    contiguous[0]["start_sec"] = 0.0
    contiguous[-1]["end_sec"] = measured_duration_sec
    # Force bookend labels when clear.
    if contiguous[0]["label"] not in ("intro", "verse"):
        contiguous[0]["label"] = "intro"
    if contiguous[-1]["label"] not in ("outro", "chorus", "verse"):
        contiguous[-1]["label"] = "outro"
    return contiguous


def sections_from_form_specs(
    file_path: str,
    specs: list[dict],
    *,
    bpm: float,
    vocals_path: str | None = None,
    meter_numerator: int = 4,
) -> list[Section]:
    """Build Section objects from snapped form specs, filling DSP features."""
    y, sr = librosa.load(file_path, sr=None, mono=True)
    beats_per_bar = max(2, int(meter_numerator))
    sections: list[Section] = []
    cumulative_bars = 0
    for index, spec in enumerate(specs):
        start = float(spec["start_sec"])
        end = float(spec["end_sec"])
        energy = _phrase_rms_energy(y, sr, start, end)
        centroid = _phrase_spectral_centroid(y, sr, start, end)
        dens = _phrase_vocal_density(
            start_sec=start,
            end_sec=end,
            vocals_path=vocals_path,
            y_mix=y,
            sr=sr,
        )
        dur = max(0.01, end - start)
        approx_beats = int(spec.get("approx_beats") or 0)
        if approx_beats <= 0 and bpm > 0:
            approx_beats = int(round(dur * (bpm / 60.0)))
        bars = max(1, int(round(approx_beats / beats_per_bar))) if approx_beats else max(
            1, int(round(dur * (bpm / 60.0) / beats_per_bar))
        )
        label = normalize_section_label(str(spec.get("label") or ""))
        if index == 0:
            label = "intro"
        if index == len(specs) - 1:
            label = "outro"
        sections.append(
            Section(
                index=index,
                start_sec=start,
                end_sec=end,
                label=label,
                energy=energy,
                bars=bars,
                spectral_centroid=centroid,
                vocal_density=dens,
                bar_start=cumulative_bars,
                bar_end=cumulative_bars + bars,
                name=str(spec.get("name") or label),
                description=str(spec.get("description") or ""),
                approx_beats=approx_beats,
            )
        )
        cumulative_bars += bars
    return sections


def _estimate_downbeat_indices(
    beat_times: np.ndarray,
    y: np.ndarray,
    sr: int,
    *,
    meter_numerator: int = 4,
) -> np.ndarray:
    """
    Estimate which beat indices are downbeats (bar starts).

    Combines onset-strength peaks at beats with a Foote-style novelty nudge, then
    picks a phase that maximizes energy on every ``meter_numerator``-th beat.
    """
    stride = max(2, int(meter_numerator))
    n = int(beat_times.size)
    if n < stride:
        return np.arange(0, n, max(n, 1), dtype=int)

    onset = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.frames_to_time(np.arange(len(onset)), sr=sr)
    beat_onset = np.interp(beat_times, onset_times, onset)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    beat_frames = librosa.time_to_frames(beat_times, sr=sr)
    beat_frames = np.unique(np.clip(beat_frames, 0, chroma.shape[1] - 1))
    if beat_frames.size >= 2:
        sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
        novelty = np.zeros(sync.shape[1], dtype=np.float64)
        for i in range(1, sync.shape[1]):
            a = sync[:, i - 1]
            b = sync[:, i]
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
            novelty[i] = 1.0 - float(np.dot(a, b) / denom)
        if novelty.size < n:
            novelty = np.pad(novelty, (0, n - novelty.size))
        novelty = novelty[:n]
    else:
        novelty = np.zeros(n, dtype=np.float64)

    score = 0.7 * (beat_onset / (np.max(beat_onset) + 1e-8)) + 0.3 * novelty
    best_phase = 0
    best_val = -np.inf
    for phase in range(stride):
        val = float(np.sum(score[phase::stride]))
        if val > best_val:
            best_val = val
            best_phase = phase
    return np.arange(best_phase, n, stride, dtype=int)


def segment_phrases(
    file_path: str,
    *,
    beats_per_phrase: int = 8,
    bars_per_phrase: int | None = None,
    max_phrases: int | None = 12,
    use_downbeats: bool = True,
    meter_numerator: int = 4,
) -> list[Phrase]:
    """
    Split audio into phrase chunks on a downbeat-aware beat grid.

    Default ``beats_per_phrase=8`` ≈ two bars in 4/4. When ``bars_per_phrase`` is
    set (e.g. 8 or 16), phrase length is ``bars_per_phrase * meter_numerator`` beats.
    """
    beats_per_bar = max(2, int(meter_numerator))
    if bars_per_phrase is not None:
        beats_per_phrase = int(bars_per_phrase) * beats_per_bar
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
        downbeats = _estimate_downbeat_indices(
            beat_times, y, sr, meter_numerator=beats_per_bar
        )
        if downbeats.size == 0:
            starts = list(range(0, n_beats - beats_per_phrase, beats_per_phrase))
        else:
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


def _phrase_spectral_centroid(
    y: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
) -> float:
    start = max(0, int(start_sec * sr))
    end = min(len(y), int(end_sec * sr))
    if end <= start + 1:
        return 0.0
    chunk = y[start:end]
    cent = librosa.feature.spectral_centroid(y=chunk, sr=sr)
    return float(np.mean(cent))


def _phrase_vocal_density(
    *,
    start_sec: float,
    end_sec: float,
    vocals_path: str | None,
    y_mix: np.ndarray,
    sr: int,
) -> float:
    """Fraction of frames with vocal activity in [start_sec, end_sec)."""
    if vocals_path:
        try:
            full = AudioSegment.from_file(vocals_path)
            start_ms = max(0, int(start_sec * 1000))
            end_ms = min(len(full), int(end_sec * 1000))
            if end_ms <= start_ms:
                return 0.0
            clip = full[start_ms:end_ms]
            mask = vocal_activity_mask(clip)
            if mask.size == 0:
                return 0.0
            return float(np.mean(mask.astype(np.float64)))
        except Exception:  # noqa: BLE001
            pass

    # Fallback: onset density on the mix as a proxy.
    start = max(0, int(start_sec * sr))
    end = min(len(y_mix), int(end_sec * sr))
    if end <= start + 1:
        return 0.0
    chunk = y_mix[start:end]
    onset = librosa.onset.onset_strength(y=chunk, sr=sr)
    if onset.size == 0:
        return 0.0
    thr = float(np.median(onset) + 0.5 * np.std(onset))
    return float(np.mean(onset > thr))


def _label_sections_rich(
    energies: list[float],
    centroids: list[float],
    densities: list[float],
) -> list[SectionLabel]:
    """
    Map multi-feature section stats to INTRO/VERSE/BUILDUP/CHORUS/DROP/OUTRO.

    Also keeps legacy low/high unused here — roles are the richer set.
    """
    n = len(energies)
    if n == 0:
        return []
    if n == 1:
        return ["chorus"]

    e = np.asarray(energies, dtype=np.float64)
    d = np.asarray(densities, dtype=np.float64)
    e_hi = float(np.quantile(e, 0.65))
    e_lo = float(np.quantile(e, 0.35))
    d_hi = float(np.quantile(d, 0.55)) if np.max(d) > 0 else 0.35

    labels: list[SectionLabel] = []
    for i in range(n):
        energy = float(e[i])
        dens = float(d[i])
        rising = i > 0 and energy > float(e[i - 1]) * 1.12

        if i == 0 and energy <= e_hi:
            labels.append("intro")
        elif i == n - 1 and energy <= e_hi:
            labels.append("outro")
        elif energy >= e_hi and dens >= d_hi:
            # Bright high-energy with vocals → chorus; very high energy → drop.
            labels.append("drop" if energy >= float(np.quantile(e, 0.8)) else "chorus")
        elif rising and dens < d_hi and energy >= e_lo:
            labels.append("buildup")
        elif dens >= d_hi * 0.6 or energy >= e_lo:
            labels.append("verse")
        else:
            labels.append("verse" if dens > 0.15 else "buildup")

    # Ensure at least one chorus/drop for pairing heuristics.
    if not any(lab in ("chorus", "drop", "high") for lab in labels):
        labels[int(np.argmax(e))] = "chorus"
    return labels


def detect_sections(
    file_path: str,
    *,
    bars_per_section: int = 8,
    max_sections: int | None = None,
    vocals_path: str | None = None,
    meter_numerator: int = 4,
    bpm: float | None = None,
) -> list[Section]:
    """
    Detect timed sections on a downbeat-aware bar grid with role labels.

    Labels use RMS energy, spectral centroid, and vocal density (from Demucs
    vocals when ``vocals_path`` is provided). Default ``max_sections=None`` keeps
    the full song (no 8-section cap). Last section is extended to song end;
    first/last labels are forced to intro/outro.
    """
    phrases = segment_phrases(
        file_path,
        bars_per_phrase=bars_per_section,
        max_phrases=max_sections,
        use_downbeats=True,
        meter_numerator=meter_numerator,
    )
    y, sr = librosa.load(file_path, sr=None, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))
    energies: list[float] = []
    centroids: list[float] = []
    densities: list[float] = []
    for phrase in phrases:
        energies.append(_phrase_rms_energy(y, sr, phrase.start_sec, phrase.end_sec))
        centroids.append(
            _phrase_spectral_centroid(y, sr, phrase.start_sec, phrase.end_sec)
        )
        densities.append(
            _phrase_vocal_density(
                start_sec=phrase.start_sec,
                end_sec=phrase.end_sec,
                vocals_path=vocals_path,
                y_mix=y,
                sr=sr,
            )
        )
    labels = _label_sections_rich(energies, centroids, densities)
    if labels:
        labels[0] = "intro"
        labels[-1] = "outro"
    beats_per_bar = max(2, int(meter_numerator))
    sections: list[Section] = []
    for phrase, energy, centroid, dens, label in zip(
        phrases, energies, centroids, densities, labels
    ):
        end_sec = phrase.end_sec
        if phrase.index == len(phrases) - 1:
            end_sec = duration
        dur = max(0.01, end_sec - phrase.start_sec)
        approx_beats = 0
        if bpm is not None and bpm > 0:
            approx_beats = int(round(dur * (bpm / 60.0)))
        else:
            approx_beats = max(1, phrase.end_beat - phrase.start_beat)
        bars = max(1, int(round(approx_beats / beats_per_bar))) if approx_beats else bars_per_section
        bar_start = phrase.index * bars_per_section
        name = str(label)
        sections.append(
            Section(
                index=phrase.index,
                start_sec=phrase.start_sec,
                end_sec=end_sec,
                label=label,
                energy=float(energy),
                bars=bars,
                spectral_centroid=float(centroid),
                vocal_density=float(dens),
                bar_start=bar_start,
                bar_end=bar_start + bars,
                name=name,
                description="",
                approx_beats=approx_beats,
            )
        )
    return sections


def section_index_for_bar(sections: list[Section], bar: int) -> int:
    """Map a bar number to the nearest section index."""
    if not sections:
        return 0
    for section in sections:
        if section.bar_start <= bar < section.bar_end:
            return section.index
    # Clamp to last section if past the end.
    if bar >= sections[-1].bar_end:
        return sections[-1].index
    return sections[0].index


def pick_section(sections: list[Section], index: int) -> Section:
    """Clamp *index* into range; empty list raises."""
    if not sections:
        raise ValueError("No sections available")
    if index < 0:
        index = 0
    if index >= len(sections):
        index = len(sections) - 1
    return sections[index]


def is_high_energy_label(label: str) -> bool:
    return label in ("chorus", "drop", "high")


def is_low_energy_label(label: str) -> bool:
    return label in ("intro", "verse", "outro", "low", "buildup")
