"""DSP helpers: BPM detection, key detection, time-stretch, pitch-shift, and mixing."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

# Pitch-class names used for key labeling (C … B).
KEY_NAMES: tuple[str, ...] = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)

# Krumhansl-Schmuckler key profiles (major / minor).
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)


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


def get_key(file_path: str) -> str:
    """
    Estimate musical key using chroma STFT + Krumhansl-Schmuckler profiles.

    Returns a label like ``\"C major\"`` or ``\"A minor\"``.
    """
    y, sr = librosa.load(file_path, sr=None, mono=True)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1).astype(np.float64)
    norm = np.linalg.norm(chroma_mean)
    if norm < 1e-8:
        raise ValueError(f"Could not detect a valid key for {file_path}")
    chroma_mean /= norm

    best_score = -np.inf
    best_key = "C major"
    major_template = _MAJOR_PROFILE / np.linalg.norm(_MAJOR_PROFILE)
    minor_template = _MINOR_PROFILE / np.linalg.norm(_MINOR_PROFILE)

    for pitch_class in range(12):
        major = np.roll(major_template, pitch_class)
        minor = np.roll(minor_template, pitch_class)
        major_score = float(np.dot(chroma_mean, major))
        minor_score = float(np.dot(chroma_mean, minor))
        root = KEY_NAMES[pitch_class]
        if major_score > best_score:
            best_score = major_score
            best_key = f"{root} major"
        if minor_score > best_score:
            best_score = minor_score
            best_key = f"{root} minor"

    return best_key


def _pitch_class_from_key(key: str) -> int:
    root = key.strip().split()[0]
    # Accept flats by normalizing a couple of common aliases.
    aliases = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
    root = aliases.get(root, root)
    try:
        return KEY_NAMES.index(root)
    except ValueError as exc:
        raise ValueError(f"Unrecognized key label: {key!r}") from exc


def semitones_to_match_key(source_key: str, target_key: str) -> int:
    """
    Shortest signed semitone shift so *source_key*'s root matches *target_key*'s root.

    Mode (major/minor) is ignored for the shift amount — we align pitch class only,
    which is the usual POC mashup approach.
    """
    diff = (_pitch_class_from_key(target_key) - _pitch_class_from_key(source_key)) % 12
    if diff > 6:
        diff -= 12
    return int(diff)


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
    sf.write(str(out), stretched, sr)
    return str(out)


def pitch_shift_audio(input_path: str, output_path: str, n_steps: float) -> str:
    """
    Pitch-shift audio by *n_steps* semitones without changing duration.

    Positive values raise pitch; negative values lower it.
    Uses ``librosa.effects.pitch_shift``.
    """
    y, sr = librosa.load(input_path, sr=None, mono=False)

    if y.ndim == 1:
        shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    else:
        channels = [
            librosa.effects.pitch_shift(y=y[i], sr=sr, n_steps=n_steps)
            for i in range(y.shape[0])
        ]
        shifted = np.stack(channels, axis=-1)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), shifted, sr)
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
