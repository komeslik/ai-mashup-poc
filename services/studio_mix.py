"""Studio grid seed + render for the Section editor."""

from __future__ import annotations

import json
import logging
import math
import shutil
import uuid
from pathlib import Path
from typing import Any, Literal

from pydub import AudioSegment

from services.audio import crossfade_concatenate, extract_audio_segment, fit_length
from services.dual_mix import CROSSFADE_MS
from services.structure import Section

logger = logging.getLogger(__name__)

STEMS = ("vocals", "drums", "bass", "other")
SongKey = Literal["a", "b"]

# Distinct colors for source sections (hex).
SECTION_PALETTE = (
    "#6ee7b7",
    "#93c5fd",
    "#f9a8d4",
    "#fcd34d",
    "#c4b5fd",
    "#fdba74",
    "#67e8f9",
    "#fca5a5",
    "#a3e635",
    "#dda0dd",
    "#7dd3fc",
    "#fde68a",
)


def _cell_key(song: str, stem: str) -> str:
    return f"{song}:{stem}"


def empty_cell() -> dict[str, Any]:
    return {"enabled": False, "source_section_index": 0}


def enrich_sections_display(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Add stable display_label (verse1, verse2, …) and color to each section dict.

    Mutates and returns the list.
    """
    counts: dict[str, int] = {}
    totals: dict[str, int] = {}
    for sec in sections:
        base = str(sec.get("label") or sec.get("name") or "section").strip().lower()
        base = "".join(ch for ch in base if ch.isalnum() or ch in ("_", "-")) or "section"
        totals[base] = totals.get(base, 0) + 1
    for i, sec in enumerate(sections):
        base = str(sec.get("label") or sec.get("name") or "section").strip().lower()
        base = "".join(ch for ch in base if ch.isalnum() or ch in ("_", "-")) or "section"
        counts[base] = counts.get(base, 0) + 1
        if totals.get(base, 1) > 1:
            display = f"{base}{counts[base]}"
        else:
            display = base
        sec["display_label"] = display
        if not sec.get("color"):
            sec["color"] = SECTION_PALETTE[i % len(SECTION_PALETTE)]
        if not sec.get("name"):
            sec["name"] = display
    return sections


def seed_studio_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Build studio.json grid from auto-mashup phrases / sections."""
    phrases = list(metadata.get("phrases") or [])
    sections_a = enrich_sections_display(list(metadata.get("sections_a") or []))
    sections_b = enrich_sections_display(list(metadata.get("sections_b") or []))
    title_a = (
        (metadata.get("form_a") or {}).get("title")
        or (metadata.get("structure_a") or {}).get("title")
        or "Song A"
    )
    title_b = (
        (metadata.get("form_b") or {}).get("title")
        or (metadata.get("structure_b") or {}).get("title")
        or "Song B"
    )

    columns: list[dict[str, Any]] = []
    for phrase in phrases:
        cells: dict[str, Any] = {
            _cell_key(song, stem): empty_cell() for song in ("a", "b") for stem in STEMS
        }
        lead = str(phrase.get("lead") or "")
        v_idx = int(phrase.get("vocal_section_index") or 0)
        i_idx = int(phrase.get("instrumental_section_index") or 0)
        # Instrumental bed = Song A drums/bass/other from instrumental section.
        for stem in ("drums", "bass", "other"):
            cells[_cell_key("a", stem)] = {
                "enabled": True,
                "source_section_index": i_idx,
            }
        if lead in ("song_a", "a"):
            cells[_cell_key("a", "vocals")] = {
                "enabled": True,
                "source_section_index": v_idx,
            }
        elif lead in ("song_b", "b"):
            cells[_cell_key("b", "vocals")] = {
                "enabled": True,
                "source_section_index": v_idx,
            }
        overlays = list(phrase.get("overlay_stems") or [])
        overlay_from = str(phrase.get("overlay_from") or "none")
        if overlay_from == "song_b":
            for stem in overlays:
                if stem in STEMS:
                    cells[_cell_key("b", stem)] = {
                        "enabled": True,
                        "source_section_index": min(
                            v_idx, max(len(sections_b) - 1, 0)
                        ),
                    }
        columns.append(
            {
                "id": f"col_{int(phrase.get('index', len(columns))):02d}_{uuid.uuid4().hex[:6]}",
                "label": phrase.get("section_name")
                or phrase.get("label")
                or f"Section {int(phrase.get('index', 0)) + 1}",
                "duration_ms": int(phrase.get("duration_ms") or 0),
                "cells": cells,
            }
        )

    if not columns:
        # One empty column covering first A section length estimate.
        dur = 8000
        if sections_a:
            dur = max(
                1000,
                int((float(sections_a[0].get("end_sec", 8)) - float(sections_a[0].get("start_sec", 0))) * 1000),
            )
        cells = {
            _cell_key(song, stem): empty_cell() for song in ("a", "b") for stem in STEMS
        }
        for stem in ("drums", "bass", "other", "vocals"):
            cells[_cell_key("a", stem)] = {"enabled": True, "source_section_index": 0}
        columns.append(
            {
                "id": f"col_00_{uuid.uuid4().hex[:6]}",
                "label": "Section 1",
                "duration_ms": dur,
                "cells": cells,
            }
        )

    return {
        "version": 1,
        "title_a": title_a,
        "title_b": title_b,
        "stems": list(STEMS),
        "columns": columns,
        "sections_a": sections_a,
        "sections_b": sections_b,
        "crossfade_ms": int(metadata.get("crossfade_ms") or CROSSFADE_MS),
        "target_bpm": metadata.get("target_bpm"),
        "beats_a": _beats_for_studio(metadata, "a"),
        "beats_b": _beats_for_studio(metadata, "b"),
        "downbeats_a": _downbeats_for_studio(metadata, "a"),
        "downbeats_b": _downbeats_for_studio(metadata, "b"),
    }


def _structure_blob(metadata: dict[str, Any], song: str) -> dict[str, Any]:
    key = f"structure_{song}"
    blob = metadata.get(key)
    return blob if isinstance(blob, dict) else {}


def _tempo_scale(metadata: dict[str, Any], song: str) -> float:
    """Map original-time → stretched-stem time (1/rate), matching dual_mix stretch."""
    struct = _structure_blob(metadata, song)
    source_bpm = float(struct.get("bpm") or struct.get("measured_bpm") or 0.0)
    target = float(metadata.get("target_bpm") or source_bpm or 0.0)
    if source_bpm <= 0 or target <= 0:
        return 1.0
    # Prefer tempo-octave candidates closest to 1.0 (same as tempo_aware_stretch_rate).
    candidates = (
        target / source_bpm,
        (target / 2.0) / source_bpm,
        (target * 2.0) / source_bpm,
    )
    rate = float(min(candidates, key=lambda r: abs(math.log2(r))))
    if rate <= 0:
        return 1.0
    return 1.0 / rate


def _scaled_times(times: list[Any], scale: float) -> list[float]:
    out: list[float] = []
    for t in times or []:
        try:
            out.append(float(t) * scale)
        except (TypeError, ValueError):
            continue
    return out


def _beats_for_studio(metadata: dict[str, Any], song: str) -> list[float]:
    struct = _structure_blob(metadata, song)
    scale = _tempo_scale(metadata, song)
    beats = _scaled_times(list(struct.get("beats") or []), scale)
    if beats:
        return beats
    # Fallback: synthetic grid from BPM on stretched timeline.
    bpm = float(struct.get("bpm") or metadata.get("target_bpm") or 0.0)
    if bpm <= 0:
        return []
    dur = 0.0
    for sec in metadata.get(f"sections_{song}") or []:
        try:
            dur = max(dur, float(sec.get("end_sec") or 0.0) * scale)
        except (TypeError, ValueError):
            continue
    if dur <= 0:
        dur = 180.0
    step = 60.0 / bpm
    t = 0.0
    out: list[float] = []
    while t <= dur + 1e-6:
        out.append(round(t, 4))
        t += step
    return out


def _downbeats_for_studio(metadata: dict[str, Any], song: str) -> list[float]:
    struct = _structure_blob(metadata, song)
    scale = _tempo_scale(metadata, song)
    return _scaled_times(list(struct.get("downbeats") or []), scale)


def load_studio(session_dir: Path) -> dict[str, Any]:
    path = session_dir / "studio.json"
    if path.is_file():
        studio = json.loads(path.read_text(encoding="utf-8"))
        studio["sections_a"] = enrich_sections_display(list(studio.get("sections_a") or []))
        studio["sections_b"] = enrich_sections_display(list(studio.get("sections_b") or []))
        # Backfill beats for older sessions from metadata when missing.
        if not studio.get("beats_a") or not studio.get("beats_b"):
            meta_path = session_dir / "metadata.json"
            if meta_path.is_file():
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                if not studio.get("beats_a"):
                    studio["beats_a"] = _beats_for_studio(metadata, "a")
                    studio["downbeats_a"] = _downbeats_for_studio(metadata, "a")
                if not studio.get("beats_b"):
                    studio["beats_b"] = _beats_for_studio(metadata, "b")
                    studio["downbeats_b"] = _downbeats_for_studio(metadata, "b")
        return studio
    meta_path = session_dir / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError("Session metadata missing")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    studio = seed_studio_from_metadata(metadata)
    save_studio(session_dir, studio)
    return studio


def save_studio(session_dir: Path, studio: dict[str, Any]) -> None:
    path = session_dir / "studio.json"
    studio = dict(studio)
    studio["sections_a"] = enrich_sections_display(list(studio.get("sections_a") or []))
    studio["sections_b"] = enrich_sections_display(list(studio.get("sections_b") or []))
    path.write_text(json.dumps(studio, indent=2), encoding="utf-8")


def ensure_song_preview(session_dir: Path, song: SongKey) -> Path:
    """Mix four stems into song_{a|b}_preview.mp3 for waveforms / audition."""
    session = Path(session_dir)
    out = session / f"song_{song}_preview.mp3"
    if out.is_file() and out.stat().st_size > 0:
        return out
    layers: list[AudioSegment] = []
    for stem in STEMS:
        try:
            path = _stem_path(session, song, stem)
        except FileNotFoundError:
            continue
        try:
            layers.append(AudioSegment.from_file(path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Preview stem load failed %s/%s: %s", song, stem, exc)
    if not layers:
        raise FileNotFoundError(f"No stems for song {song} preview")
    canvas = layers[0]
    for layer in layers[1:]:
        # Overlay at equal gain; pad shorter layers implicitly via overlay.
        if len(layer) > len(canvas):
            canvas = canvas + AudioSegment.silent(
                duration=len(layer) - len(canvas), frame_rate=canvas.frame_rate
            )
        canvas = canvas.overlay(layer)
    canvas.export(str(out), format="mp3")
    return out


def extract_song_range(
    session_dir: Path,
    song: SongKey,
    start_sec: float,
    end_sec: float,
    output_path: Path,
) -> Path:
    """Export a time range of the song preview (builds preview if needed)."""
    preview = ensure_song_preview(session_dir, song)
    start = max(0.0, float(start_sec))
    end = max(start + 0.05, float(end_sec))
    seg = AudioSegment.from_file(preview)
    start_ms = int(start * 1000)
    end_ms = min(len(seg), int(end * 1000))
    clip = seg[start_ms:end_ms]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip.export(str(output_path), format="mp3")
    return output_path


def apply_committed_sections(
    studio: dict[str, Any],
    *,
    sections_a: list[dict[str, Any]] | None = None,
    sections_b: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Replace source section maps and rescale mashup column durations when a
    column's Song A bed section length changed.
    """
    old_a = list(studio.get("sections_a") or [])
    old_b = list(studio.get("sections_b") or [])
    if sections_a is not None:
        studio["sections_a"] = enrich_sections_display(list(sections_a))
    else:
        studio["sections_a"] = enrich_sections_display(old_a)
    if sections_b is not None:
        studio["sections_b"] = enrich_sections_display(list(sections_b))
    else:
        studio["sections_b"] = enrich_sections_display(old_b)

    new_a = studio["sections_a"]
    new_b = studio["sections_b"]

    def _dur(secs: list[dict[str, Any]], idx: int) -> float:
        if not secs:
            return 0.0
        sec = secs[int(idx) % len(secs)]
        return max(0.0, float(sec.get("end_sec", 0)) - float(sec.get("start_sec", 0)))

    for column in studio.get("columns") or []:
        cells = column.get("cells") or {}
        # Prefer Song A drums bed as length reference (matches seed).
        bed = cells.get(_cell_key("a", "drums")) or {}
        if not bed.get("enabled"):
            bed = cells.get(_cell_key("a", "other")) or bed
        if not bed.get("enabled"):
            continue
        idx = int(bed.get("source_section_index") or 0)
        old_len = _dur(old_a, idx)
        new_len = _dur(new_a, idx)
        if old_len > 0.05 and new_len > 0.05:
            scale = new_len / old_len
            cur = int(column.get("duration_ms") or 0)
            if cur > 0:
                column["duration_ms"] = max(250, int(round(cur * scale)))
            else:
                column["duration_ms"] = max(250, int(round(new_len * 1000)))
    return studio


def _section_window(sections: list[dict[str, Any]], index: int) -> tuple[float, float]:
    if not sections:
        return 0.0, 8.0
    idx = int(index) % len(sections)
    sec = sections[idx]
    start = float(sec.get("start_sec", 0.0))
    end = float(sec.get("end_sec", start + 8.0))
    if end <= start:
        end = start + 0.5
    return start, end


def _cell_source_duration_ms(sections: list[dict[str, Any]], index: int) -> int:
    start, end = _section_window(sections, index)
    return max(250, int(round((end - start) * 1000)))


def column_mix_duration_ms(
    column: dict[str, Any],
    sections_a: list[dict[str, Any]],
    sections_b: list[dict[str, Any]],
) -> int:
    """
    Column length for mixing.

    If ``duration_lock_key`` points at an enabled cell, use that cell's source
    section length only (no max-of-all / no looping fill).

    Otherwise: max(stored duration, longest enabled source section).
    """
    cells = column.get("cells") or {}
    sections_map = {"a": sections_a, "b": sections_b}
    lock_key = column.get("duration_lock_key")
    if isinstance(lock_key, str) and lock_key:
        locked = cells.get(lock_key)
        if locked and locked.get("enabled"):
            song = "a" if lock_key.startswith("a:") else "b"
            return _cell_source_duration_ms(
                sections_map[song], int(locked.get("source_section_index") or 0)
            )

    lengths: list[int] = []
    for song in ("a", "b"):
        for stem in STEMS:
            cell = cells.get(_cell_key(song, stem)) or cells.get(f"{song}_{stem}")
            if not cell or not cell.get("enabled"):
                continue
            lengths.append(
                _cell_source_duration_ms(
                    sections_map[song], int(cell.get("source_section_index") or 0)
                )
            )
    stored = int(column.get("duration_ms") or 0)
    if lengths:
        natural = max(lengths)
        return max(stored, natural) if stored > 0 else natural
    return stored or 4000


def _stem_path(session_dir: Path, song: str, stem: str) -> Path:
    folder = session_dir / f"stems_{song}"
    for ext in ("wav", "mp3"):
        path = folder / f"{stem}.{ext}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing stem {song}/{stem} under {folder}")


def render_studio(
    session_dir: str | Path,
    studio: dict[str, Any] | None = None,
    *,
    column_id: str | None = None,
    output_path: str | Path | None = None,
) -> str:
    """
    Mix the studio grid to MP3.

    If *column_id* is set, render only that column; otherwise concatenate all.
    """
    session = Path(session_dir)
    if studio is None:
        studio = load_studio(session)
    sections_a = list(studio.get("sections_a") or [])
    sections_b = list(studio.get("sections_b") or [])
    sections_map = {"a": sections_a, "b": sections_b}
    fade_ms = int(studio.get("crossfade_ms") or CROSSFADE_MS)

    columns = list(studio.get("columns") or [])
    if column_id:
        columns = [c for c in columns if c.get("id") == column_id]
        if not columns:
            raise ValueError(f"Unknown column_id: {column_id}")

    mixed: list[AudioSegment] = []
    work = session / "_studio_work"
    work.mkdir(exist_ok=True)

    for col_i, column in enumerate(columns):
        cells = column.get("cells") or {}
        frame_rate = 44100
        channels = 2
        layers: list[AudioSegment] = []

        for song in ("a", "b"):
            for stem in STEMS:
                cell = cells.get(_cell_key(song, stem)) or cells.get(f"{song}_{stem}")
                if not cell or not cell.get("enabled"):
                    continue
                src_idx = int(cell.get("source_section_index") or 0)
                start, end = _section_window(sections_map[song], src_idx)
                stem_file = _stem_path(session, song, stem)
                clip_path = work / f"{col_i}_{song}_{stem}.wav"
                try:
                    extract_audio_segment(
                        str(stem_file),
                        str(clip_path),
                        start,
                        end,
                    )
                    seg = AudioSegment.from_file(clip_path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Studio cell %s/%s failed: %s", song, stem, exc)
                    continue
                frame_rate = seg.frame_rate or frame_rate
                channels = seg.channels or channels
                layers.append(seg)

        # Duration: locked cell wins; otherwise max of enabled sources (loop fill).
        lock_key = column.get("duration_lock_key")
        locked = bool(
            isinstance(lock_key, str)
            and lock_key
            and (cells.get(lock_key) or {}).get("enabled")
        )
        duration_ms = column_mix_duration_ms(column, sections_a, sections_b)
        if layers and not locked:
            duration_ms = max(duration_ms, max(len(s) for s in layers))
        if duration_ms <= 0:
            duration_ms = 4000
        # Keep studio.json / UI in sync with effective mix length.
        column["duration_ms"] = int(duration_ms)

        canvas = AudioSegment.silent(duration=duration_ms, frame_rate=frame_rate).set_channels(
            channels
        )
        for seg in layers:
            # Locked column: trim/pad only — never loop shorter stems.
            fitted = fit_length(seg, duration_ms, loop=not locked)
            canvas = canvas.overlay(fitted)
        mixed.append(canvas)

    if not mixed:
        raise RuntimeError("Studio grid produced no audio (enable at least one cell)")

    if len(mixed) == 1:
        combined = mixed[0]
    else:
        combined = crossfade_concatenate(mixed, crossfade_ms=min(fade_ms, 200))

    out = Path(output_path) if output_path else (session / "mashup-edit.mp3")
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(out), format="mp3")
    # Persist any duration_ms updates from max-source logic.
    if studio is not None and not column_id:
        try:
            save_studio(session, studio)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not persist updated column durations: %s", exc)
    return str(out)


def persist_aligned_stems(
    session_dir: Path,
    *,
    stems_a: dict[str, str],
    stems_b: dict[str, str],
) -> None:
    """Copy pitch/BPM-aligned stem files into the session for studio use."""
    for song, mapping in (("a", stems_a), ("b", stems_b)):
        dest = session_dir / f"stems_{song}"
        dest.mkdir(parents=True, exist_ok=True)
        for name, src in mapping.items():
            if not src:
                continue
            src_path = Path(src)
            if not src_path.is_file():
                continue
            target = dest / f"{name}{src_path.suffix.lower() or '.wav'}"
            if src_path.resolve() != target.resolve():
                shutil.copy2(src_path, target)
    # Invalidate / rebuild song previews after stem copy.
    for song in ("a", "b"):
        preview = session_dir / f"song_{song}_preview.mp3"
        if preview.is_file():
            preview.unlink(missing_ok=True)
        try:
            ensure_song_preview(session_dir, song)  # type: ignore[arg-type]
        except FileNotFoundError:
            logger.warning("Could not build song_%s_preview", song)


def sections_from_prompt_dicts(rows: list[dict[str, Any]]) -> list[Section]:
    """Rebuild lightweight Section list from metadata prompt dicts (unused helper)."""
    out: list[Section] = []
    for i, row in enumerate(rows):
        out.append(
            Section(
                index=int(row.get("index", i)),
                start_sec=float(row.get("start_sec", 0.0)),
                end_sec=float(row.get("end_sec", 0.0)),
                label=row.get("label") or "other",  # type: ignore[arg-type]
                energy=float(row.get("energy") or 0.0),
                bars=int(row.get("bars") or 0),
                spectral_centroid=float(row.get("spectral_centroid") or 0.0),
                vocal_density=float(row.get("vocal_density") or 0.0),
                bar_start=int(row.get("bar_start") or 0),
                bar_end=int(row.get("bar_end") or 0),
                name=str(row.get("name") or ""),
                description=str(row.get("description") or ""),
                approx_beats=int(row.get("approx_beats") or 0),
            )
        )
    return out
