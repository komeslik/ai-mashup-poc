"""DSP helpers: BPM detection, time-stretching, and stem mixing."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError


def get_bpm(file_path: str) -> float:
    """Estimate tempo (BPM) of an audio file using librosa beat tracking."""
    y, sr = librosa.load(file_path, sr=None, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    if isinstance(tempo, np.ndarray):
        tempo_value = float(tempo[0]) if tempo.size else 0.0
    else:
        tempo_value = float(tempo)

    if tempo_value <= 0:
        raise ValueError(f"Could not detect a valid BPM for {file_path}")

    return tempo_value


def time_stretch_audio(input_path: str, output_path: str, rate: float) -> str:
    """
    Time-stretch audio by *rate* without changing pitch.

    ``rate > 1`` speeds up (higher BPM); ``rate < 1`` slows down.
    Uses ``librosa.effects.time_stretch`` for POC reliability.
    """
    if rate <= 0:
        raise ValueError(f"stretch rate must be positive, got {rate}")

    y, sr = librosa.load(input_path, sr=None, mono=False)

    # librosa.effects.time_stretch expects 1-D mono; handle stereo by channel.
    if y.ndim == 1:
        stretched = librosa.effects.time_stretch(y=y, rate=rate)
    else:
        channels = [
            librosa.effects.time_stretch(y=y[i], rate=rate) for i in range(y.shape[0])
        ]
        # soundfile expects (samples, channels)
        stretched = np.stack(channels, axis=-1)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if stretched.ndim == 1:
        sf.write(str(out), stretched, sr)
    else:
        sf.write(str(out), stretched, sr)

    return str(out)


def mix_stems(vocal_path: str, instrumental_path: str, output_path: str) -> str:
    """Overlay vocals onto instrumental and export an MP3 mashup."""
    try:
        vocals = AudioSegment.from_file(vocal_path)
        instrumental = AudioSegment.from_file(instrumental_path)
    except CouldntDecodeError as exc:
        raise RuntimeError(
            "Failed to decode audio. Ensure ffmpeg is installed and on PATH."
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg appears to be missing. Install ffmpeg to enable MP3 export."
        ) from exc

    # Match sample rates / channels for a clean overlay.
    if vocals.frame_rate != instrumental.frame_rate:
        vocals = vocals.set_frame_rate(instrumental.frame_rate)
    if vocals.channels != instrumental.channels:
        vocals = vocals.set_channels(instrumental.channels)

    duration_ms = max(len(vocals), len(instrumental))
    canvas = AudioSegment.silent(duration=duration_ms, frame_rate=instrumental.frame_rate)
    canvas = canvas.set_channels(instrumental.channels)
    mixed = canvas.overlay(instrumental).overlay(vocals)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        mixed.export(str(out), format="mp3")
    except Exception as exc:  # noqa: BLE001 — pydub/ffmpeg errors vary
        raise RuntimeError(
            "Failed to export MP3. Ensure ffmpeg is installed and on PATH."
        ) from exc

    return str(out)
