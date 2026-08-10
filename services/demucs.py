"""Local stem separation via the Demucs CLI."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

DEMUCS_MODEL = "htdemucs"


class DemucsError(Exception):
    """Raised when local Demucs stem separation fails."""


def separate_stems(input_audio_path: str, output_dir: str) -> tuple[str, str]:
    """
    Separate an audio file into vocals and instrumental stems using Demucs CLI.

    Runs::

        demucs -n htdemucs --two-stems vocals --mp3 -o {output_dir} {input_audio_path}

    ``--mp3`` avoids torchaudio's torchcodec WAV writer (broken on newer torchaudio).
    Demucs writes::

        {output_dir}/htdemucs/{song_name}/vocals.mp3
        {output_dir}/htdemucs/{song_name}/no_vocals.mp3

    Returns
    -------
    tuple[str, str]
        ``(vocals_path, instrumental_path)`` as absolute paths.
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
    logger.info("Running Demucs: %s", " ".join(command))

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        if completed.stderr:
            # Demucs/tqdm progress goes to stderr even on success.
            logger.debug("Demucs stderr: %s", completed.stderr[-2000:])
    except FileNotFoundError as exc:
        raise DemucsError(
            "demucs CLI not found. Install with: pip install demucs"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        # Prefer the real exception line over tqdm progress spam.
        detail = stderr or stdout or str(exc)
        for marker in ("ImportError:", "ModuleNotFoundError:", "Error:", "Traceback"):
            idx = detail.rfind(marker)
            if idx != -1:
                detail = detail[idx:]
                break
        raise DemucsError(f"Demucs CLI failed: {detail[-4000:]}") from exc

    song_name = audio_file.stem
    stem_dir = out / DEMUCS_MODEL / song_name
    vocals_path = stem_dir / "vocals.mp3"
    instrumental_path = stem_dir / "no_vocals.mp3"

    if not vocals_path.is_file():
        raise DemucsError(f"Expected vocals stem missing at {vocals_path}")
    if not instrumental_path.is_file():
        raise DemucsError(f"Expected instrumental stem missing at {instrumental_path}")

    return str(vocals_path.resolve()), str(instrumental_path.resolve())
