"""Studio grid seed + render for the Section editor."""

from __future__ import annotations

import json
import logging
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


def _cell_key(song: str, stem: str) -> str:
    return f"{song}:{stem}"


def empty_cell() -> dict[str, Any]:
    return {"enabled": False, "source_section_index": 0}


def seed_studio_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Build studio.json grid from auto-mashup phrases / sections."""
    phrases = list(metadata.get("phrases") or [])
    sections_a = list(metadata.get("sections_a") or [])
    sections_b = list(metadata.get("sections_b") or [])
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
    }


def load_studio(session_dir: Path) -> dict[str, Any]:
    path = session_dir / "studio.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    meta_path = session_dir / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError("Session metadata missing")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    studio = seed_studio_from_metadata(metadata)
    save_studio(session_dir, studio)
    return studio


def save_studio(session_dir: Path, studio: dict[str, Any]) -> None:
    path = session_dir / "studio.json"
    path.write_text(json.dumps(studio, indent=2), encoding="utf-8")


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
        duration_ms = int(column.get("duration_ms") or 0)
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

        if duration_ms <= 0:
            duration_ms = max((len(s) for s in layers), default=4000)
        canvas = AudioSegment.silent(duration=duration_ms, frame_rate=frame_rate).set_channels(
            channels
        )
        for seg in layers:
            fitted = fit_length(seg, duration_ms)
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
