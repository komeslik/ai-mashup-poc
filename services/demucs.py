"""Local stem separation via the Demucs CLI."""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEMUCS_MODEL = "htdemucs"
STEM_NAMES = ("vocals", "drums", "bass", "other")


class DemucsError(Exception):
    """Raised when local Demucs stem separation fails."""


@dataclass(frozen=True)
class FullStems:
    """Four-stem Demucs output plus a combined instrumental bed."""

    vocals: str
    drums: str
    bass: str
    other: str
    instrumental: str  # no_vocals equivalent (drums+bass+other mix path or synthesized)
    stem_dir: str = ""  # folder containing the four stems (WAV)


def separate_stems(input_audio_path: str, output_dir: str) -> tuple[str, str]:
    """
    Separate an audio file into vocals and instrumental stems (two-stem mode).

    Returns ``(vocals_path, instrumental_path)``.
    """
    audio_file = Path(input_audio_path)
    if not audio_file.is_file():
        raise DemucsError(f"Audio file not found: {input_audio_path}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    command = [
        "demucs",
        "-n",
        DEMUCS_MODEL,
        "--two-stems",
        "vocals",
        "--mp3",
        "-o",
        str(out),
        str(audio_file),
    ]
    _run_demucs(command)

    song_name = audio_file.stem
    stem_dir = out / DEMUCS_MODEL / song_name
    vocals_path = stem_dir / "vocals.mp3"
    instrumental_path = stem_dir / "no_vocals.mp3"

    if not vocals_path.is_file():
        raise DemucsError(f"Expected vocals stem missing at {vocals_path}")
    if not instrumental_path.is_file():
        raise DemucsError(f"Expected instrumental stem missing at {instrumental_path}")

    return str(vocals_path.resolve()), str(instrumental_path.resolve())


def separate_full_stems(
    input_audio_path: str,
    output_dir: str,
    *,
    wav: bool = True,
) -> FullStems:
    """
    Separate into vocals / drums / bass / other (full htdemucs).

    By default emits **WAV** stems (allin1 demix-cache compatible). Builds
    ``instrumental`` by overlaying drums+bass+other when ``no_vocals`` is absent.
    """
    from pydub import AudioSegment

    audio_file = Path(input_audio_path)
    if not audio_file.is_file():
        raise DemucsError(f"Audio file not found: {input_audio_path}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    command = [
        "demucs",
        "-n",
        DEMUCS_MODEL,
        "-o",
        str(out),
        str(audio_file),
    ]
    if not wav:
        command.insert(-2, "--mp3")
    _run_demucs(command)

    song_name = audio_file.stem
    stem_dir = out / DEMUCS_MODEL / song_name
    ext = "wav" if wav else "mp3"
    vocals = stem_dir / f"vocals.{ext}"
    drums = stem_dir / f"drums.{ext}"
    bass = stem_dir / f"bass.{ext}"
    other = stem_dir / f"other.{ext}"

    for path in (vocals, drums, bass, other):
        if not path.is_file():
            raise DemucsError(f"Expected stem missing at {path}")

    instrumental_path = stem_dir / f"no_vocals.{ext}"
    if not instrumental_path.is_file():
        drums_seg = AudioSegment.from_file(drums)
        bass_seg = AudioSegment.from_file(bass)
        other_seg = AudioSegment.from_file(other)
        if bass_seg.frame_rate != drums_seg.frame_rate:
            bass_seg = bass_seg.set_frame_rate(drums_seg.frame_rate)
        if other_seg.frame_rate != drums_seg.frame_rate:
            other_seg = other_seg.set_frame_rate(drums_seg.frame_rate)
        duration = max(len(drums_seg), len(bass_seg), len(other_seg))
        canvas = AudioSegment.silent(duration=duration, frame_rate=drums_seg.frame_rate)
        canvas = canvas.set_channels(drums_seg.channels)
        bed = canvas.overlay(drums_seg).overlay(bass_seg).overlay(other_seg)
        bed.export(str(instrumental_path), format=ext)

    return FullStems(
        vocals=str(vocals.resolve()),
        drums=str(drums.resolve()),
        bass=str(bass.resolve()),
        other=str(other.resolve()),
        instrumental=str(instrumental_path.resolve()),
        stem_dir=str(stem_dir.resolve()),
    )


def ensure_allin1_demix_layout(
    demix_dir: str | Path,
    track_stem: str,
    stems: FullStems,
) -> Path:
    """
    Place four WAV stems at ``{demix_dir}/htdemucs/{track_stem}/*.wav`` for allin1.

    If ``stems`` already live in that folder, this is a no-op. Otherwise copies.
    """
    root = Path(demix_dir)
    dest = root / DEMUCS_MODEL / track_stem
    dest.mkdir(parents=True, exist_ok=True)
    mapping = {
        "vocals": stems.vocals,
        "drums": stems.drums,
        "bass": stems.bass,
        "other": stems.other,
    }
    for name, src in mapping.items():
        target = dest / f"{name}.wav"
        if target.is_file():
            continue
        src_path = Path(src)
        if src_path.resolve() == target.resolve():
            continue
        if src_path.suffix.lower() == ".wav":
            shutil.copy2(src_path, target)
        else:
            from pydub import AudioSegment

            AudioSegment.from_file(src_path).export(str(target), format="wav")
    return dest


def _run_demucs(command: list[str]) -> None:
    logger.info("Running Demucs: %s", " ".join(command))
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        if completed.stderr:
            logger.debug("Demucs stderr: %s", completed.stderr[-2000:])
    except FileNotFoundError as exc:
        raise DemucsError(
            "demucs CLI not found. Install with: pip install demucs"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or str(exc)
        for marker in ("ImportError:", "ModuleNotFoundError:", "Error:", "Traceback"):
            idx = detail.rfind(marker)
            if idx != -1:
                detail = detail[idx:]
                break
        raise DemucsError(f"Demucs CLI failed: {detail[-4000:]}") from exc
