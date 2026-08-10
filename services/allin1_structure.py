"""Song structure via allin1 (real audio timestamps) with DSP fallback."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from services.structure import (
    Section,
    detect_sections,
    normalize_section_label,
    sections_from_form_specs,
)

logger = logging.getLogger(__name__)

# allin1 Harmonix labels → our SectionLabel vocabulary
_ALLIN1_LABEL_MAP: dict[str, str] = {
    "start": "intro",
    "end": "outro",
    "intro": "intro",
    "outro": "outro",
    "verse": "verse",
    "chorus": "chorus",
    "bridge": "bridge",
    "break": "bridge",
    "inst": "other",
    "solo": "bridge",
}


@dataclass
class StructureBundle:
    """Normalized structure analysis for one song."""

    sections: list[Section]
    bpm: float
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)
    beat_positions: list[int] = field(default_factory=list)
    source: str = "dsp_fallback"  # allin1 | dsp_fallback
    meter_numerator: int = 4
    raw_segments: list[dict[str, Any]] = field(default_factory=list)

    def metadata(self) -> dict[str, Any]:
        return {
            "structure_source": self.source,
            "bpm": self.bpm,
            "meter_numerator": self.meter_numerator,
            "meter_denominator": 4,
            "n_beats": len(self.beats),
            "n_downbeats": len(self.downbeats),
            "beats": self.beats,
            "downbeats": self.downbeats,
            "beat_positions": self.beat_positions,
            "segments": self.raw_segments,
            "sections": [s.to_prompt_dict() for s in self.sections],
        }


def _ensure_wav(path: str, work_dir: Path | None = None) -> str:
    """
    Prefer WAV for allin1 (MP3 decoder offsets can skew beats).

    Returns path to a WAV file (original if already wav, else converted copy).
    """
    src = Path(path)
    if src.suffix.lower() in (".wav", ".wave"):
        return str(src.resolve())

    out_dir = work_dir or Path(tempfile.mkdtemp(prefix="allin1_wav_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{src.stem}.wav"
    if out.is_file():
        return str(out.resolve())

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("ffmpeg not found; passing %s to allin1 as-is", src.name)
        return str(src.resolve())

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-ac",
        "2",
        "-ar",
        "44100",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return str(out.resolve())
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "WAV convert failed (%s); using original for allin1",
            (exc.stderr or "")[-500:],
        )
        return str(src.resolve())


def map_allin1_label(raw: str) -> str:
    key = (raw or "").strip().lower()
    if key in _ALLIN1_LABEL_MAP:
        return _ALLIN1_LABEL_MAP[key]
    return normalize_section_label(key)


def _meter_from_beat_positions(positions: list[int]) -> int:
    if not positions:
        return 4
    m = max(int(p) for p in positions if p is not None)
    return max(2, min(m, 12))


def _segments_to_specs(
    segments: list[Any],
    *,
    bpm: float,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for seg in segments:
        if hasattr(seg, "start"):
            start = float(seg.start)
            end = float(seg.end)
            label_raw = str(getattr(seg, "label", "verse"))
        else:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            label_raw = str(seg.get("label", "verse"))
        if end <= start + 0.05:
            continue
        # Drop tiny allin1 "start"/"end" markers under 0.5s when adjacent sections exist
        label = map_allin1_label(label_raw)
        approx_beats = int(round((end - start) * (bpm / 60.0))) if bpm > 0 else 0
        specs.append(
            {
                "name": label_raw,
                "label": label,
                "start_sec": start,
                "end_sec": end,
                "approx_beats": approx_beats,
                "description": f"allin1:{label_raw}",
            }
        )
    return specs


def _merge_tiny_edge_markers(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold sub-0.6s start/end markers into neighbors."""
    if len(specs) < 2:
        return specs
    out: list[dict[str, Any]] = []
    for i, spec in enumerate(specs):
        dur = float(spec["end_sec"]) - float(spec["start_sec"])
        name = str(spec.get("name") or "").lower()
        if dur < 0.6 and name in ("start", "end") and out:
            # extend previous
            prev = out[-1]
            prev["end_sec"] = spec["end_sec"]
            continue
        if dur < 0.6 and name == "start" and i + 1 < len(specs):
            nxt = dict(specs[i + 1])
            nxt["start_sec"] = spec["start_sec"]
            # skip current; next will be rewritten when we reach it — handle by
            # absorbing into next via mutating lookahead
            specs[i + 1] = nxt
            continue
        out.append(dict(spec))
    return out or specs


def analyze_structure(
    file_path: str,
    *,
    vocals_path: str | None = None,
    work_dir: str | Path | None = None,
    fallback_bpm: float | None = None,
    demix_dir: str | Path | None = None,
) -> StructureBundle:
    """
    Run allin1 on *file_path*; fall back to librosa ``detect_sections`` on failure.

    When *demix_dir* already contains ``htdemucs/{stem}/*.wav`` matching the
    analyze input basename, allin1 skips its internal Demucs pass.
    """
    import librosa

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio not found: {file_path}")

    work = Path(work_dir) if work_dir else None
    wav_path = _ensure_wav(str(path), work)

    try:
        import allin1

        logger.info("allin1.analyze %s", path.name)
        cache_root = work or Path(tempfile.mkdtemp(prefix="allin1_cache_"))
        cache_root.mkdir(parents=True, exist_ok=True)
        resolved_demix = Path(demix_dir) if demix_dir else (cache_root / "demix")
        resolved_demix.mkdir(parents=True, exist_ok=True)
        # multiprocess=False: Pool workers break under uvicorn / macOS spawn.
        result = allin1.analyze(
            wav_path,
            device="cpu",
            multiprocess=False,
            demix_dir=str(resolved_demix),
            spec_dir=str(cache_root / "spec"),
            keep_byproducts=bool(work) or demix_dir is not None,
        )
        bpm = float(getattr(result, "bpm", 0.0) or 0.0)
        if bpm <= 0 and fallback_bpm:
            bpm = float(fallback_bpm)
        if bpm <= 0:
            y_tmp, sr_tmp = librosa.load(wav_path, sr=None, mono=True)
            bpm = float(librosa.beat.tempo(y=y_tmp, sr=sr_tmp)[0])

        beats = [float(b) for b in (getattr(result, "beats", None) or [])]
        downbeats = [float(b) for b in (getattr(result, "downbeats", None) or [])]
        beat_positions = [int(p) for p in (getattr(result, "beat_positions", None) or [])]
        meter = _meter_from_beat_positions(beat_positions)

        raw_segments = []
        for seg in getattr(result, "segments", None) or []:
            raw_segments.append(
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "label": str(seg.label),
                }
            )

        specs = _merge_tiny_edge_markers(_segments_to_specs(result.segments, bpm=bpm))
        if not specs:
            raise RuntimeError("allin1 returned no usable segments")

        # Ensure full coverage of measured duration.
        duration = float(librosa.get_duration(path=wav_path))
        specs[0]["start_sec"] = 0.0
        specs[-1]["end_sec"] = duration
        specs[0]["label"] = "intro"
        specs[-1]["label"] = "outro"

        sections = sections_from_form_specs(
            str(path),
            specs,
            bpm=bpm,
            vocals_path=vocals_path,
            meter_numerator=meter,
        )
        if not sections:
            raise RuntimeError("allin1 segments produced empty Section list")

        return StructureBundle(
            sections=sections,
            bpm=bpm,
            beats=beats,
            downbeats=downbeats,
            beat_positions=beat_positions,
            source="allin1",
            meter_numerator=meter,
            raw_segments=raw_segments,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("allin1 failed for %s (%s); DSP detect_sections fallback", path.name, exc)
        bpm = float(fallback_bpm or 0.0)
        if bpm <= 0:
            try:
                from services.audio import get_bpm

                bpm = float(get_bpm(str(path)))
            except Exception:  # noqa: BLE001
                bpm = 120.0
        sections = detect_sections(
            str(path),
            vocals_path=vocals_path,
            max_sections=None,
            meter_numerator=4,
            bpm=bpm,
        )
        return StructureBundle(
            sections=sections,
            bpm=bpm,
            source="dsp_fallback",
            meter_numerator=4,
        )


def resolve_sections_for_song(
    file_path: str,
    title: str,
    bpm: float,
    vocals_path: str | None = None,
    *,
    measured_duration_sec: float | None = None,
    work_dir: str | Path | None = None,
    demix_dir: str | Path | None = None,
    **_ignored: Any,
) -> tuple[list[Section], dict[str, Any] | None, dict[str, Any]]:
    """
    allin1 structure resolver.

    Returns (sections, form_dict_or_none, metadata).
    ``form_dict`` mirrors a lightweight summary for UI.
    Pass *demix_dir* with precomputed WAV stems to skip allin1's Demucs.
    """
    del measured_duration_sec  # allin1 uses the file itself
    bundle = analyze_structure(
        file_path,
        vocals_path=vocals_path,
        work_dir=work_dir,
        fallback_bpm=bpm,
        demix_dir=demix_dir,
    )
    form_summary: dict[str, Any] | None = {
        "title": title,
        "bpm": bundle.bpm,
        "time_signature_numerator": bundle.meter_numerator,
        "time_signature_denominator": 4,
        "style_notes": f"structure via {bundle.source}",
        "total_duration_sec": (
            bundle.sections[-1].end_sec if bundle.sections else 0.0
        ),
        "sections": [
            {
                "name": s.name or s.label,
                "label": s.label,
                "start_sec": s.start_sec,
                "end_sec": s.end_sec,
                "approx_beats": s.approx_beats,
                "description": s.description,
            }
            for s in bundle.sections
        ],
    }
    meta = bundle.metadata()
    meta["title"] = title
    meta["measured_bpm"] = bpm
    meta["source"] = bundle.source  # back-compat with older UI keys
    return bundle.sections, form_summary, meta
