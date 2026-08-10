"""Local audio library listing and mashability ranking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from services.audio import get_bpm
from services.mashability import MashabilityWeights, find_best_mashability_alignment

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".aiff", ".aif"}


@dataclass(frozen=True)
class LibraryTrack:
    id: str
    name: str
    path: str
    bpm: float | None = None


def library_root(base_dir: Path) -> Path:
    root = base_dir / "library"
    root.mkdir(parents=True, exist_ok=True)
    return root


def list_library_tracks(base_dir: Path) -> list[LibraryTrack]:
    root = library_root(base_dir)
    tracks: list[LibraryTrack] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        rel = path.relative_to(root).as_posix()
        bpm: float | None
        try:
            bpm = float(get_bpm(str(path)))
        except Exception:  # noqa: BLE001
            bpm = None
        tracks.append(
            LibraryTrack(
                id=rel,
                name=path.stem,
                path=str(path.resolve()),
                bpm=bpm,
            )
        )
    return tracks


def rank_library_against_query(
    query_path: str,
    base_dir: Path,
    *,
    top_k: int = 5,
    weights: MashabilityWeights | None = None,
) -> list[dict]:
    """
    Rank library tracks by local mashability against a query audio file.

    Used for paper-style collection search (and future Album mode).
    """
    tracks = list_library_tracks(base_dir)
    scored: list[dict] = []
    for track in tracks:
        if Path(track.path).resolve() == Path(query_path).resolve():
            continue
        try:
            alignment = find_best_mashability_alignment(
                query_path,
                track.path,
                weights=weights,
                window_beats=16,
            )
            scored.append(
                {
                    "id": track.id,
                    "name": track.name,
                    "path": track.path,
                    "bpm": track.bpm,
                    "score": float(alignment.score),
                    "harmonic_score": float(alignment.harmonic_score),
                    "rhythmic_score": float(alignment.rhythmic_score),
                    "spectral_score": float(alignment.spectral_score),
                    "n_steps": int(alignment.n_steps),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Library rank failed for %s: %s", track.id, exc)

    scored.sort(key=lambda row: row["score"], reverse=True)
    return scored[: max(1, top_k)]
