"""DSP helpers: BPM, beats, key/chroma mashability, stretch, pitch, loudness mix."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

logger = logging.getLogger(__name__)

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

# Default harmonic search window (~8 bars in 4/4).
DEFAULT_WINDOW_BEATS = 32
DEFAULT_MAX_KEY_SHIFT = 6


@dataclass(frozen=True)
class HarmonicAlignment:
    """Best local harmonic match between a vocal stem and an instrumental stem."""

    n_steps: int
    score: float
    window_beats: int
    vocal_beat_start: int
    instrumental_beat_start: int
    vocal_start_sec: float
    vocal_end_sec: float
    instrumental_start_sec: float
    instrumental_end_sec: float


def get_bpm(file_path: str) -> float:
    """Estimate tempo (BPM) of an audio file using librosa beat tracking."""
    tempo, _ = get_beats(file_path)
    return tempo


def get_beats(file_path: str) -> tuple[float, np.ndarray]:
    """
    Return ``(tempo_bpm, beat_times_sec)`` for an audio file.

    Beat times are seconds from the start of the file.
    """
    y, sr = librosa.load(file_path, sr=None, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    if isinstance(tempo, np.ndarray):
        tempo_value = float(tempo[0]) if tempo.size else 0.0
    else:
        tempo_value = float(tempo)

    duration = float(librosa.get_duration(y=y, sr=sr))
    if tempo_value <= 0:
        # Fall back for pathological / nearly silent inputs.
        tempo_value = 120.0

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    if beat_times.size < 2:
        # Synthetic grid so downstream chroma sync still works.
        step = 60.0 / tempo_value
        beat_times = np.arange(0.0, max(duration, step), step, dtype=np.float64)

    return tempo_value, np.asarray(beat_times, dtype=np.float64)


def tempo_aware_stretch_rate(source_bpm: float, target_bpm: float) -> float:
    """
    Stretch rate so ``source_bpm`` lands on ``target_bpm``, preferring tempo octaves.

    If tempos are near a 2× relationship, prefer a rate closer to 1.0 (half/double
    feel) over an extreme stretch — AutoMashUpper §III tempo-octave handling.
    """
    if source_bpm <= 0 or target_bpm <= 0:
        raise ValueError("BPM values must be positive")

    candidates = (
        target_bpm / source_bpm,
        (target_bpm / 2.0) / source_bpm,
        (target_bpm * 2.0) / source_bpm,
    )
    return float(min(candidates, key=lambda rate: abs(math.log2(rate))))


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


def _beat_sync_chroma(y: np.ndarray, sr: int, beat_times: np.ndarray) -> np.ndarray:
    """Return a ``(12, n_beats)`` beat-synchronous chromagram."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    beat_frames = librosa.time_to_frames(beat_times, sr=sr)
    # Ensure strictly increasing frame boundaries for sync.
    beat_frames = np.unique(np.clip(beat_frames, 0, chroma.shape[1] - 1))
    if beat_frames.size < 2:
        return np.mean(chroma, axis=1, keepdims=True).astype(np.float64)
    synced = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
    return np.asarray(synced, dtype=np.float64)


def _normalize_chroma_columns(chroma: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return chroma / norms


def _window_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two ``(12, T)`` chroma patches."""
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    denom = float(np.linalg.norm(flat_a) * np.linalg.norm(flat_b))
    if denom < 1e-8:
        return -1.0
    return float(np.dot(flat_a, flat_b) / denom)


def find_best_harmonic_alignment(
    vocal_path: str,
    instrumental_path: str,
    *,
    window_beats: int = DEFAULT_WINDOW_BEATS,
    max_key_shift: int = DEFAULT_MAX_KEY_SHIFT,
) -> HarmonicAlignment:
    """
    Search local beat offset + key rotation for best vocal↔instrumental chroma match.

    Returns the best window boundaries (seconds) and pitch shift in semitones.
    """
    y_v, sr_v = librosa.load(vocal_path, sr=None, mono=True)
    y_i, sr_i = librosa.load(instrumental_path, sr=None, mono=True)

    _, beats_v = get_beats(vocal_path)
    _, beats_i = get_beats(instrumental_path)

    chroma_v = _normalize_chroma_columns(_beat_sync_chroma(y_v, sr_v, beats_v))
    chroma_i = _normalize_chroma_columns(_beat_sync_chroma(y_i, sr_i, beats_i))

    n_v = chroma_v.shape[1]
    n_i = chroma_i.shape[1]
    win = min(window_beats, n_v, n_i)
    if win < 2:
        # Degenerate: fall back to whole-file key match on a tiny window.
        duration_v = float(librosa.get_duration(y=y_v, sr=sr_v))
        duration_i = float(librosa.get_duration(y=y_i, sr=sr_i))
        try:
            n_steps = semitones_to_match_key(get_key(vocal_path), get_key(instrumental_path))
        except ValueError:
            n_steps = 0
        end = min(duration_v, duration_i)
        return HarmonicAlignment(
            n_steps=n_steps,
            score=0.0,
            window_beats=max(win, 1),
            vocal_beat_start=0,
            instrumental_beat_start=0,
            vocal_start_sec=0.0,
            vocal_end_sec=end,
            instrumental_start_sec=0.0,
            instrumental_end_sec=end,
        )

    key_shifts = list(range(-max_key_shift, max_key_shift + 1))
    best_score = -np.inf
    best: tuple[int, int, int] = (0, 0, 0)  # n_steps, vocal_start, instr_start

    # Subsample vocal starts a bit for speed on long tracks.
    vocal_step = 1 if n_v - win < 48 else max(1, (n_v - win) // 24)
    instr_step = 1 if n_i - win < 64 else max(1, (n_i - win) // 32)

    for n_steps in key_shifts:
        rotated = np.roll(chroma_v, n_steps, axis=0)
        for v0 in range(0, n_v - win + 1, vocal_step):
            patch_v = rotated[:, v0 : v0 + win]
            for i0 in range(0, n_i - win + 1, instr_step):
                patch_i = chroma_i[:, i0 : i0 + win]
                score = _window_correlation(patch_v, patch_i)
                if score > best_score:
                    best_score = score
                    best = (n_steps, v0, i0)

    n_steps, v0, i0 = best
    v1 = v0 + win
    i1 = i0 + win

    # Map beat indices to times; last beat index uses file end if needed.
    duration_v = float(librosa.get_duration(y=y_v, sr=sr_v))
    duration_i = float(librosa.get_duration(y=y_i, sr=sr_i))

    def _beat_span(beats: np.ndarray, start: int, end: int, duration: float) -> tuple[float, float]:
        t0 = float(beats[start]) if start < len(beats) else 0.0
        if end < len(beats):
            t1 = float(beats[end])
        else:
            t1 = duration
        if t1 <= t0:
            t1 = min(duration, t0 + (60.0 / 120.0) * max(end - start, 1))
        return t0, t1

    v_start, v_end = _beat_span(beats_v, v0, v1, duration_v)
    i_start, i_end = _beat_span(beats_i, i0, i1, duration_i)

    return HarmonicAlignment(
        n_steps=int(n_steps),
        score=float(best_score),
        window_beats=int(win),
        vocal_beat_start=int(v0),
        instrumental_beat_start=int(i0),
        vocal_start_sec=v_start,
        vocal_end_sec=v_end,
        instrumental_start_sec=i_start,
        instrumental_end_sec=i_end,
    )


def extract_audio_segment(
    input_path: str,
    output_path: str,
    start_sec: float,
    end_sec: float,
) -> str:
    """Write ``[start_sec, end_sec)`` of *input_path* to *output_path* (WAV)."""
    if end_sec <= start_sec:
        raise ValueError(f"Invalid segment bounds: {start_sec} .. {end_sec}")

    y, sr = librosa.load(input_path, sr=None, mono=False)
    start = max(0, int(start_sec * sr))
    end = int(end_sec * sr)

    if y.ndim == 1:
        end = min(end, y.shape[0])
        segment = y[start:end]
    else:
        end = min(end, y.shape[1])
        segment = y[:, start:end]
        segment = np.ascontiguousarray(segment.T)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), segment, sr)
    return str(out)


def time_stretch_audio(input_path: str, output_path: str, rate: float) -> str:
    """
    Time-stretch audio by *rate* without changing pitch.

    Prefers Rubber Band (``pyrubberband`` / ffmpeg) when available; falls back to librosa.
    ``rate > 1`` speeds up (higher BPM); ``rate < 1`` slows down.
    """
    if rate <= 0:
        raise ValueError(f"stretch rate must be positive, got {rate}")
    if abs(rate - 1.0) < 1e-6:
        # Copy via load/write so callers always get a WAV they can mutate.
        y, sr = librosa.load(input_path, sr=None, mono=False)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if y.ndim == 1:
            sf.write(str(out), y, sr)
        else:
            sf.write(str(out), np.ascontiguousarray(y.T), sr)
        return str(out)

    y, sr = librosa.load(input_path, sr=None, mono=False)
    stretched = _rubberband_stretch(y, sr, rate)
    if stretched is None:
        if y.ndim == 1:
            stretched = librosa.effects.time_stretch(y=y, rate=rate)
        else:
            channels = [
                librosa.effects.time_stretch(y=y[i], rate=rate) for i in range(y.shape[0])
            ]
            stretched = np.stack(channels, axis=-1)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(stretched, np.ndarray) and stretched.ndim == 2 and stretched.shape[0] in (1, 2):
        # Channel-first from pyrubberband path — write as (samples, channels).
        if stretched.shape[0] <= 2 and stretched.shape[1] > 2:
            sf.write(str(out), np.ascontiguousarray(stretched.T), sr)
        else:
            sf.write(str(out), stretched, sr)
    else:
        sf.write(str(out), stretched, sr)
    return str(out)


def _rubberband_stretch(y: np.ndarray, sr: int, rate: float) -> np.ndarray | None:
    """Try pyrubberband time-stretch; return None if unavailable."""
    try:
        import pyrubberband as pyrb  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    try:
        if y.ndim == 1:
            return pyrb.time_stretch(y, sr, rate)
        channels = [pyrb.time_stretch(y[i], sr, rate) for i in range(y.shape[0])]
        return np.stack(channels, axis=-1)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pyrubberband time_stretch failed (%s); using librosa", exc)
        return None


def pitch_shift_audio(input_path: str, output_path: str, n_steps: float) -> str:
    """
    Pitch-shift audio by *n_steps* semitones without changing duration.

    Prefers Rubber Band when available; falls back to librosa.
    """
    y, sr = librosa.load(input_path, sr=None, mono=False)
    shifted = _rubberband_pitch(y, sr, n_steps)
    if shifted is None:
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
    if isinstance(shifted, np.ndarray) and shifted.ndim == 2 and shifted.shape[0] in (1, 2):
        if shifted.shape[0] <= 2 and shifted.shape[1] > 2:
            sf.write(str(out), np.ascontiguousarray(shifted.T), sr)
        else:
            sf.write(str(out), shifted, sr)
    else:
        sf.write(str(out), shifted, sr)
    return str(out)


def _rubberband_pitch(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray | None:
    try:
        import pyrubberband as pyrb  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    try:
        if y.ndim == 1:
            return pyrb.pitch_shift(y, sr, n_steps)
        channels = [pyrb.pitch_shift(y[i], sr, n_steps) for i in range(y.shape[0])]
        return np.stack(channels, axis=-1)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pyrubberband pitch_shift failed (%s); using librosa", exc)
        return None


def estimate_tuning_cents(file_path: str) -> float:
    """Estimate global tuning offset in cents (100 cents = 1 semitone)."""
    y, sr = librosa.load(file_path, sr=None, mono=True)
    try:
        tuning = float(librosa.estimate_tuning(y=y, sr=sr))
    except Exception:  # noqa: BLE001
        return 0.0
    # librosa returns fractional semitones; convert to cents.
    return float(tuning * 100.0)


def pitch_shift_with_cents(
    input_path: str,
    output_path: str,
    n_steps: float,
    *,
    apply_tuning_correction: bool = True,
) -> tuple[str, float]:
    """
    Pitch-shift by integer/fractional semitones, optionally refining with tuning cents.

    Returns ``(output_path, cents_applied)``.
    """
    cents = 0.0
    total_steps = float(n_steps)
    if apply_tuning_correction:
        cents = estimate_tuning_cents(input_path)
        # Nudge toward concert pitch / remove residual mistune (clamped).
        cents = max(-50.0, min(50.0, -cents))
        total_steps += cents / 100.0
    path = pitch_shift_audio(input_path, output_path, total_steps)
    return path, cents


def highpass_audio(
    input_path: str,
    output_path: str,
    *,
    cutoff_hz: float = 100.0,
) -> str:
    """Apply a gentle high-pass filter (SciPy butter via librosa/scipy)."""
    from scipy.signal import butter, sosfilt

    y, sr = librosa.load(input_path, sr=None, mono=False)
    nyq = 0.5 * sr
    normal = min(0.99, max(0.001, cutoff_hz / nyq))
    sos = butter(2, normal, btype="highpass", output="sos")

    if y.ndim == 1:
        filtered = sosfilt(sos, y)
    else:
        filtered = np.stack([sosfilt(sos, y[i]) for i in range(y.shape[0])], axis=-1)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), filtered if filtered.ndim == 1 else filtered, sr)
    return str(out)


def highpass_segment(segment: AudioSegment, *, cutoff_hz: float = 100.0) -> AudioSegment:
    """High-pass a pydub segment via temporary WAV round-trip."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="hpf_") as tmp:
        src = Path(tmp) / "in.wav"
        dst = Path(tmp) / "out.wav"
        segment.export(str(src), format="wav")
        highpass_audio(str(src), str(dst), cutoff_hz=cutoff_hz)
        return AudioSegment.from_file(str(dst))


def _segment_rms_db(segment: AudioSegment) -> float:
    """RMS level in dBFS; silence returns a very low floor."""
    rms = segment.rms
    if rms <= 0:
        return -60.0
    return float(segment.dBFS)


def match_loudness_lufs(
    source: AudioSegment,
    reference: AudioSegment,
    *,
    target_offset_db: float = -1.5,
) -> AudioSegment:
    """
    Match *source* integrated loudness (LUFS) toward *reference* using pyloudnorm.

    Falls back to RMS matching if pyloudnorm is unavailable.
    """
    try:
        import pyloudnorm as pyln
    except Exception:  # noqa: BLE001
        return match_loudness_rms(source, reference, target_offset_db=target_offset_db)

    def _to_float_mono(seg: AudioSegment) -> tuple[np.ndarray, int]:
        samples = np.array(seg.get_array_of_samples(), dtype=np.float64)
        max_amp = float(1 << (8 * seg.sample_width - 1))
        samples = samples / max_amp
        if seg.channels > 1:
            samples = samples.reshape((-1, seg.channels))
        else:
            samples = samples.reshape((-1, 1))
        return samples, seg.frame_rate

    try:
        src_audio, sr = _to_float_mono(source)
        ref_audio, _ = _to_float_mono(reference)
        meter = pyln.Meter(sr)
        src_lufs = float(meter.integrated_loudness(src_audio))
        ref_lufs = float(meter.integrated_loudness(ref_audio))
        if not np.isfinite(src_lufs) or not np.isfinite(ref_lufs):
            return match_loudness_rms(source, reference, target_offset_db=target_offset_db)
        delta = (ref_lufs + target_offset_db) - src_lufs
        delta = max(-18.0, min(18.0, delta))
        return source.apply_gain(delta)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LUFS match failed (%s); falling back to RMS", exc)
        return match_loudness_rms(source, reference, target_offset_db=target_offset_db)


def match_loudness_rms(
    source: AudioSegment,
    reference: AudioSegment,
    *,
    target_offset_db: float = -1.5,
) -> AudioSegment:
    """Gain-match *source* toward *reference* RMS."""
    src_db = _segment_rms_db(source)
    ref_db = _segment_rms_db(reference)
    if src_db <= -59.0:
        return source
    delta = (ref_db + target_offset_db) - src_db
    delta = max(-18.0, min(18.0, delta))
    return source.apply_gain(delta)


def match_loudness(
    source: AudioSegment,
    reference: AudioSegment,
    *,
    target_offset_db: float = -1.5,
) -> AudioSegment:
    """Loudness match preferring LUFS, with RMS fallback."""
    return match_loudness_lufs(source, reference, target_offset_db=target_offset_db)


def fit_length(segment: AudioSegment, target_ms: int) -> AudioSegment:
    """Trim or loop *segment* so its duration equals *target_ms*."""
    if target_ms <= 0:
        return segment
    if len(segment) == target_ms:
        return segment
    if len(segment) > target_ms:
        return segment[:target_ms]
    if len(segment) == 0:
        return AudioSegment.silent(duration=target_ms, frame_rate=segment.frame_rate)

    pieces: list[AudioSegment] = []
    filled = 0
    while filled < target_ms:
        pieces.append(segment)
        filled += len(segment)
    combined = sum(pieces[1:], pieces[0])
    return combined[:target_ms]


def mix_stems(
    vocal_path: str,
    instrumental_path: str,
    output_path: str,
    *,
    match_vocal_loudness: bool = True,
    trim_to_instrumental: bool = True,
    highpass_vocals_hz: float | None = 100.0,
) -> str:
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

    if highpass_vocals_hz is not None and highpass_vocals_hz > 0:
        try:
            vocals = highpass_segment(vocals, cutoff_hz=highpass_vocals_hz)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vocal high-pass failed (%s); continuing without", exc)

    if trim_to_instrumental:
        target_ms = len(instrumental)
        vocals = fit_length(vocals, target_ms)
        duration_ms = target_ms
    else:
        duration_ms = max(len(vocals), len(instrumental))
        instrumental = fit_length(instrumental, duration_ms)
        vocals = fit_length(vocals, duration_ms)

    if match_vocal_loudness:
        vocals = match_loudness(vocals, instrumental)

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
