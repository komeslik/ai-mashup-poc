"""Section-aware dual-vocal (and bassline) mashup assembly."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydub import AudioSegment

from services.agent import ArrangementSegment, MashupBlueprint
from services.audio import (
    crossfade_concatenate,
    crossfade_concatenate_adaptive,
    extract_audio_segment,
    fit_length,
    get_key,
    highpass_segment,
    match_loudness,
    pitch_shift_with_cents,
    semitones_to_match_key,
    tempo_aware_stretch_rate,
    time_stretch_audio,
)
from services.mashability import (
    LOW_RHYTHM_THRESHOLD,
    MashabilityWeights,
    find_best_mashability_alignment,
)
from services.structure import Section, detect_sections, is_high_energy_label, pick_section
from services.vad import apply_overlap_mute, duck_instrumental_under_vocals

logger = logging.getLogger(__name__)

PhraseVocalPolicy = Literal["alternate", "a_lead_b_harmony", "b_lead_a_harmony"]
CreativeMode = Literal["forced_match", "style_contrast", "bassline"]

HARMONY_GAIN_DB = -7.0
STRETCH_EPSILON = 0.01
VOCAL_HPF_HZ = 120.0
INSTR_DUCK_DB = -2.5
CROSSFADE_MS = 600
CROSSFADE_LEAD_CHANGE_MS = 100
CROSSFADE_BOOKEND_MS = 50
DEFAULT_BARS_PER_SECTION = 8
DEFAULT_MAX_SECTIONS = 8


@dataclass
class PhraseScheduleEntry:
    index: int
    lead: str
    start_sec: float
    end_sec: float
    duration_ms: int
    mashability_score: float
    harmonic_score: float
    rhythmic_score: float
    spectral_score: float
    n_steps: int
    cents: float
    harmony: bool
    harmony_suppressed_low_rhythm: bool
    vad_overlap_frames: int
    vad_ducked_frames: int
    vad_muted_frames: int = 0
    instr_ducked_frames: int = 0
    enabled: bool = True
    phrase_file: str | None = None
    section_name: str | None = None
    vocal_section_index: int | None = None
    instrumental_section_index: int | None = None
    label: str | None = None
    high_pass_filter: bool = True
    vocal_volume_db: float = 0.0
    overlay_stems: list[str] = field(default_factory=list)
    overlay_from: str = "none"


@dataclass
class MashupResult:
    output_path: str
    session_dir: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _stretch_to_bpm(
    vocal_path: str,
    output_path: str,
    source_bpm: float,
    target_bpm: float,
) -> str:
    rate = tempo_aware_stretch_rate(source_bpm, target_bpm)
    if abs(rate - 1.0) < STRETCH_EPSILON:
        return vocal_path
    return time_stretch_audio(vocal_path, output_path, rate)


def _scale_sections_after_stretch(
    sections: list[Section],
    source_bpm: float,
    target_bpm: float,
) -> list[Section]:
    """Map original-time section bounds onto a tempo-stretched stem."""
    rate = tempo_aware_stretch_rate(source_bpm, target_bpm)
    if abs(rate - 1.0) < STRETCH_EPSILON:
        return sections
    # time_stretch rate > 1 speeds up → durations shrink by 1/rate.
    scale = 1.0 / rate
    return [
        Section(
            index=s.index,
            start_sec=s.start_sec * scale,
            end_sec=s.end_sec * scale,
            label=s.label,
            energy=s.energy,
            bars=s.bars,
            spectral_centroid=s.spectral_centroid,
            vocal_density=s.vocal_density,
            bar_start=s.bar_start,
            bar_end=s.bar_end,
            name=s.name,
            description=s.description,
            approx_beats=s.approx_beats,
        )
        for s in sections
    ]


def _key_steps(vocal_path: str, instrumental_path: str) -> int:
    try:
        return int(semitones_to_match_key(get_key(vocal_path), get_key(instrumental_path)))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Key detect failed (%s); n_steps=0", exc)
        return 0


def _prepare_contiguous_lead(
    vocal_path: str,
    vocal_section: Section,
    instrumental_phrase_path: str,
    work_dir: Path,
    tag: str,
    weights: MashabilityWeights | None = None,
    *,
    fixed_n_steps: int | None = None,
    already_pitched: bool = False,
) -> tuple[AudioSegment, dict[str, float | int | bool]]:
    """
    Slice a contiguous vocal section; score against the instrumental section.

    When *fixed_n_steps* is set (global key meeting), mashability only searches
    windows — pitch is locked. If *already_pitched*, skip pitch shift (n_steps=0
    for DSP; metadata still reports the planned shift).
    """
    vocal_clip = extract_audio_segment(
        vocal_path,
        str(work_dir / f"{tag}_section.wav"),
        vocal_section.start_sec,
        vocal_section.end_sec,
    )
    try:
        alignment = find_best_mashability_alignment(
            vocal_clip,
            instrumental_phrase_path,
            weights=weights,
            window_beats=32,
            fixed_n_steps=0 if already_pitched else fixed_n_steps,
        )
        n_steps = (
            int(fixed_n_steps)
            if fixed_n_steps is not None
            else int(alignment.n_steps)
        )
        scores: dict[str, float | int | bool] = {
            "mashability_score": float(alignment.score),
            "harmonic_score": float(alignment.harmonic_score),
            "rhythmic_score": float(alignment.rhythmic_score),
            "spectral_score": float(alignment.spectral_score),
            "n_steps": n_steps,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Section mashability failed (%s); using fixed/key match", exc)
        if fixed_n_steps is not None:
            n_steps = int(fixed_n_steps)
        else:
            n_steps = _key_steps(vocal_clip, instrumental_phrase_path)
        scores = {
            "mashability_score": 0.0,
            "harmonic_score": 0.0,
            "rhythmic_score": 1.0,
            "spectral_score": 0.0,
            "n_steps": n_steps,
        }

    logger.info(
        "Section vocal %s — contiguous [%.1f–%.1f] score=%.3f n_steps=%+d pitched=%s",
        tag,
        vocal_section.start_sec,
        vocal_section.end_sec,
        scores["mashability_score"],
        scores["n_steps"],
        already_pitched,
    )
    cents = 0.0
    apply_steps = 0.0 if already_pitched else float(scores["n_steps"])
    pitched, cents = pitch_shift_with_cents(
        vocal_clip,
        str(work_dir / f"{tag}_pitched.wav"),
        apply_steps,
        apply_tuning_correction=True,
    )
    scores["cents"] = float(cents)
    return AudioSegment.from_file(pitched), scores


def _pitch_shift_stem(
    path: str,
    output_path: str,
    n_steps: int,
) -> str:
    if n_steps == 0:
        return path
    pitched, _ = pitch_shift_with_cents(
        path,
        output_path,
        float(n_steps),
        apply_tuning_correction=False,
    )
    return pitched


def _lead_letter(vocal_source: str) -> Literal["a", "b"] | None:
    if vocal_source == "song_a":
        return "a"
    if vocal_source == "song_b":
        return "b"
    return None


def _adaptive_fade_ms_list(lead_sequence: list[str | None]) -> list[int]:
    """
    Per-join fades: 100ms on lead change, 600ms same lead;
    50ms on first and last bookend joins when possible.
    """
    n = len(lead_sequence)
    if n <= 1:
        return []
    fades: list[int] = []
    for i in range(n - 1):
        prev = lead_sequence[i]
        nxt = lead_sequence[i + 1]
        if prev != nxt:
            fades.append(CROSSFADE_LEAD_CHANGE_MS)
        else:
            fades.append(CROSSFADE_MS)
    # Lighter bookend joins (intro→next and penultimate→outro).
    if fades:
        fades[0] = min(fades[0], CROSSFADE_BOOKEND_MS)
    if len(fades) > 1:
        fades[-1] = min(fades[-1], CROSSFADE_BOOKEND_MS)
    return fades


def _allows_harmony(policy: PhraseVocalPolicy) -> bool:
    return policy in ("a_lead_b_harmony", "b_lead_a_harmony")


def _match_format(seg: AudioSegment, reference: AudioSegment) -> AudioSegment:
    if seg.frame_rate != reference.frame_rate:
        seg = seg.set_frame_rate(reference.frame_rate)
    if seg.channels != reference.channels:
        seg = seg.set_channels(reference.channels)
    return seg


def _heuristic_timeline(
    sections_instr: list[Section],
    sections_a: list[Section],
    sections_b: list[Section],
    policy: PhraseVocalPolicy,
    first_lead: Literal["a", "b"],
) -> list[ArrangementSegment]:
    """Build an alternate-lead timeline when no LLM blueprint is supplied."""
    timeline: list[ArrangementSegment] = []
    for i, instr in enumerate(sections_instr):
        if policy == "a_lead_b_harmony":
            lead: Literal["song_a", "song_b"] = "song_a"
            harmony = True
        elif policy == "b_lead_a_harmony":
            lead = "song_b"
            harmony = True
        else:
            if first_lead == "a":
                lead = "song_a" if i % 2 == 0 else "song_b"
            else:
                lead = "song_b" if i % 2 == 0 else "song_a"
            harmony = False
        vocal_sections = sections_a if lead == "song_a" else sections_b
        # Prefer matching label, else same index, else first.
        match = next((s for s in vocal_sections if s.label == instr.label), None)
        if match is None:
            match = pick_section(vocal_sections, min(i, len(vocal_sections) - 1))
        timeline.append(
            ArrangementSegment(
                section_name=f"{lead} {match.label} over instr {instr.label}",
                vocal_source=lead,
                vocal_section_index=match.index,
                instrumental_section_index=instr.index,
                harmony=harmony,
            )
        )
    return timeline


def build_dual_vocal_mashup(
    *,
    vocals_a: str,
    vocals_b: str,
    instrumental: str,
    bpm_a: float,
    bpm_b: float,
    target_bpm: float,
    work_dir: Path,
    output_path: str,
    policy: PhraseVocalPolicy = "alternate",
    first_lead: Literal["a", "b"] = "a",
    bars_per_section: int = DEFAULT_BARS_PER_SECTION,
    max_sections: int | None = DEFAULT_MAX_SECTIONS,
    weights: MashabilityWeights | None = None,
    creative_mode: CreativeMode = "forced_match",
    session_dir: Path | None = None,
    blueprint: MashupBlueprint | None = None,
    sections_a: list[Section] | None = None,
    sections_b: list[Section] | None = None,
    # Global key-meeting shifts (applied once to stems before section mix).
    shift_a: int = 0,
    shift_b: int = 0,
    key_a: str | None = None,
    key_b: str | None = None,
    meeting_pc: int | None = None,
    # Optional Song B overlay stems.
    drums_b: str | None = None,
    bass_b: str | None = None,
    other_b: str | None = None,
    # Structure analysis metadata (optional; allin1 or DSP fallback).
    form_a: dict[str, Any] | None = None,
    form_b: dict[str, Any] | None = None,
    structure_meta_a: dict[str, Any] | None = None,
    structure_meta_b: dict[str, Any] | None = None,
    meter_a: int | None = None,
    meter_b: int | None = None,
    # Back-compat aliases from older call sites.
    beats_per_phrase: int = 8,
    bars_per_phrase: int | None = None,
    max_phrases: int | None = None,
) -> MashupResult:
    """
    Build a mashup from contiguous section clips (macro structure).

    Song A is the anchor bed. When *blueprint* is provided, its timeline drives
    vocal/instrumental section picks and selective Song B stem overlays.
    """
    del beats_per_phrase  # unused; kept for call-site compatibility
    if bars_per_phrase is not None:
        bars_per_section = bars_per_phrase
    if max_phrases is not None:
        max_sections = max_phrases
    # Fallback detect_sections always uncapped; max_sections kept for API compat.
    _ = max_sections

    work_dir.mkdir(parents=True, exist_ok=True)
    if session_dir is None:
        session_dir = work_dir / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    phrases_dir = session_dir / "phrases"
    phrases_dir.mkdir(exist_ok=True)

    stretch_bpm = target_bpm
    if creative_mode == "style_contrast":
        stretch_bpm = float(np_clip_tempo(target_bpm))

    # Pitch-shift full stems once (minimal meeting key), then stretch to target BPM.
    pitched_instr = _pitch_shift_stem(
        instrumental, str(work_dir / "instr_pitched.wav"), shift_a
    )
    pitched_voc_a = _pitch_shift_stem(
        vocals_a, str(work_dir / "vocals_a_pitched.wav"), shift_a
    )
    pitched_voc_b = _pitch_shift_stem(
        vocals_b, str(work_dir / "vocals_b_pitched.wav"), shift_b
    )
    overlay_paths: dict[str, str] = {}
    if drums_b:
        overlay_paths["drums"] = _pitch_shift_stem(
            drums_b, str(work_dir / "drums_b_pitched.wav"), shift_b
        )
    if bass_b:
        overlay_paths["bass"] = _pitch_shift_stem(
            bass_b, str(work_dir / "bass_b_pitched.wav"), shift_b
        )
    if other_b:
        overlay_paths["other"] = _pitch_shift_stem(
            other_b, str(work_dir / "other_b_pitched.wav"), shift_b
        )

    stretched_instr = _stretch_to_bpm(
        pitched_instr,
        str(work_dir / "instr_stretched.wav"),
        bpm_a,
        stretch_bpm,
    )
    stretched_a = _stretch_to_bpm(
        pitched_voc_a,
        str(work_dir / "vocals_a_stretched.wav"),
        bpm_a,
        stretch_bpm,
    )
    stretched_b = _stretch_to_bpm(
        pitched_voc_b,
        str(work_dir / "vocals_b_stretched.wav"),
        bpm_b,
        stretch_bpm,
    )
    stretched_overlays: dict[str, str] = {}
    for name, path_stem in overlay_paths.items():
        stretched_overlays[name] = _stretch_to_bpm(
            path_stem,
            str(work_dir / f"{name}_b_stretched.wav"),
            bpm_b,
            stretch_bpm,
        )

    # Fallback DSP path: uncapped sections (no max-8).
    if sections_a is None:
        sections_a = detect_sections(
            vocals_a,
            bars_per_section=bars_per_section,
            max_sections=None,
            meter_numerator=meter_a or 4,
            bpm=bpm_a,
        )
    if sections_b is None:
        sections_b = detect_sections(
            vocals_b,
            bars_per_section=bars_per_section,
            max_sections=None,
            meter_numerator=meter_b or 4,
            bpm=bpm_b,
        )

    # Anchor: always use Song A section map for the instrumental bed.
    sections_instr = list(sections_a)

    sections_a_stretched = _scale_sections_after_stretch(sections_a, bpm_a, stretch_bpm)
    sections_b_stretched = _scale_sections_after_stretch(sections_b, bpm_b, stretch_bpm)
    sections_instr_stretched = _scale_sections_after_stretch(
        sections_instr, bpm_a, stretch_bpm
    )

    if blueprint is not None:
        timeline = list(blueprint.timeline)
    else:
        timeline = _heuristic_timeline(
            sections_instr, sections_a, sections_b, policy, first_lead
        )

    logger.info(
        "Section mashup: %d segments, policy=%s, mode=%s, shifts A%+d B%+d, overlays=%s",
        len(timeline),
        policy,
        creative_mode,
        shift_a,
        shift_b,
        list(stretched_overlays.keys()),
    )

    mixed_phrases: list[AudioSegment] = []
    schedule: list[PhraseScheduleEntry] = []
    instrumental = stretched_instr
    lead_sequence: list[str | None] = []

    for index, segment in enumerate(timeline):
        instr_section = pick_section(
            sections_instr_stretched, segment.instrumental_section_index
        )
        instr_phrase = extract_audio_segment(
            instrumental,
            str(work_dir / f"instr_section_{index}.wav"),
            instr_section.start_sec,
            instr_section.end_sec,
        )
        instrumental_seg = AudioSegment.from_file(instr_phrase)
        target_ms = len(instrumental_seg)

        lead = _lead_letter(segment.vocal_source)
        harmony = bool(segment.harmony) and _allows_harmony(policy)
        if not _allows_harmony(policy):
            harmony = False

        canvas = AudioSegment.silent(
            duration=target_ms,
            frame_rate=instrumental_seg.frame_rate,
        ).set_channels(instrumental_seg.channels)

        scores: dict[str, float | int | bool] = {
            "mashability_score": 0.0,
            "harmonic_score": 0.0,
            "rhythmic_score": 1.0,
            "spectral_score": 0.0,
            "n_steps": shift_a if lead == "a" else (shift_b if lead == "b" else 0),
            "cents": 0.0,
        }
        lead_seg: AudioSegment | None = None
        apply_hpf = bool(getattr(segment, "high_pass_filter", True))
        vocal_gain = float(getattr(segment, "vocal_volume_db", 0.0) or 0.0)
        instr_ducked_frames = 0
        overlay_stems = list(getattr(segment, "overlay_stems", []) or [])
        overlay_from = getattr(segment, "overlay_from", "none") or "none"
        overlay_gain = float(getattr(segment, "overlay_volume_db", -6.0) or -6.0)

        # Anti-bleed: never layer Demucs "other" under an active lead vocal.
        if lead is not None:
            overlay_stems = [s for s in overlay_stems if s != "other"]
            if not overlay_stems:
                overlay_from = "none"

        if lead is not None:
            if lead == "a":
                lead_path = stretched_a
                other_path = stretched_b
                vocal_map = sections_a_stretched
                other_map = sections_b_stretched
                lead_shift = shift_a
            else:
                lead_path = stretched_b
                other_path = stretched_a
                vocal_map = sections_b_stretched
                other_map = sections_a_stretched
                lead_shift = shift_b

            vocal_section_used = pick_section(vocal_map, segment.vocal_section_index)
            lead_seg, scores = _prepare_contiguous_lead(
                lead_path,
                vocal_section_used,
                instr_phrase,
                work_dir,
                tag=f"s{index}_lead_{lead}",
                weights=weights,
                fixed_n_steps=lead_shift,
                already_pitched=True,
            )
            scores["n_steps"] = lead_shift
            lead_seg = fit_length(lead_seg, target_ms)
            lead_seg = _match_format(lead_seg, instrumental_seg)
            if apply_hpf:
                try:
                    lead_seg = highpass_segment(lead_seg, cutoff_hz=VOCAL_HPF_HZ)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Lead HPF failed: %s", exc)
            lead_seg = match_loudness(lead_seg, instrumental_seg)
            if abs(vocal_gain) > 0.01:
                lead_seg = lead_seg.apply_gain(vocal_gain)

            try:
                ducked = duck_instrumental_under_vocals(
                    instrumental_seg,
                    lead_seg,
                    duck_db=INSTR_DUCK_DB,
                )
                instrumental_seg = ducked.segment
                instr_ducked_frames = ducked.ducked_frames
            except Exception as exc:  # noqa: BLE001
                logger.warning("Instrumental duck failed: %s", exc)

        if overlay_from == "song_b" and overlay_stems and stretched_overlays:
            b_idx = (
                segment.vocal_section_index
                if segment.vocal_source == "song_b"
                else min(index, max(len(sections_b_stretched) - 1, 0))
            )
            b_sec = pick_section(sections_b_stretched, b_idx)
            for stem_name in overlay_stems:
                stem_path = stretched_overlays.get(stem_name)
                if not stem_path:
                    continue
                try:
                    clip = extract_audio_segment(
                        stem_path,
                        str(work_dir / f"overlay_{stem_name}_{index}.wav"),
                        b_sec.start_sec,
                        b_sec.end_sec,
                    )
                    overlay_seg = AudioSegment.from_file(clip)
                    overlay_seg = fit_length(overlay_seg, target_ms)
                    overlay_seg = _match_format(overlay_seg, instrumental_seg)
                    overlay_seg = overlay_seg.apply_gain(overlay_gain)
                    instrumental_seg = instrumental_seg.overlay(overlay_seg)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Overlay %s failed on segment %d: %s", stem_name, index, exc
                    )

        phrase_mix = canvas.overlay(instrumental_seg)
        if lead_seg is not None:
            phrase_mix = phrase_mix.overlay(lead_seg)

        harmony_applied = False
        harmony_suppressed = False
        vad_overlap = 0
        vad_ducked = 0
        vad_muted = 0

        if lead is not None and harmony and lead_seg is not None:
            if float(scores["rhythmic_score"]) < LOW_RHYTHM_THRESHOLD:
                logger.warning(
                    "Segment %d low rhythm score %.3f — suppressing harmony",
                    index,
                    scores["rhythmic_score"],
                )
                harmony_suppressed = True
            else:
                try:
                    other_section = pick_section(
                        other_map,
                        segment.vocal_section_index
                        if segment.vocal_section_index < len(other_map)
                        else 0,
                    )
                    high_other = next(
                        (s for s in other_map if is_high_energy_label(s.label)),
                        other_section,
                    )
                    other_shift = shift_b if lead == "a" else shift_a
                    harmony_seg, _ = _prepare_contiguous_lead(
                        other_path,
                        high_other,
                        instr_phrase,
                        work_dir,
                        tag=f"s{index}_harm",
                        weights=weights,
                        fixed_n_steps=other_shift,
                        already_pitched=True,
                    )
                    harmony_seg = fit_length(harmony_seg, target_ms)
                    harmony_seg = _match_format(harmony_seg, instrumental_seg)
                    if apply_hpf:
                        try:
                            harmony_seg = highpass_segment(
                                harmony_seg, cutoff_hz=VOCAL_HPF_HZ
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Harmony HPF failed: %s", exc)
                    harmony_seg = match_loudness(
                        harmony_seg,
                        instrumental_seg,
                        target_offset_db=HARMONY_GAIN_DB,
                    )
                    vad = apply_overlap_mute(lead_seg, harmony_seg)
                    harmony_seg = vad.segment
                    vad_overlap = vad.overlap_frames
                    vad_ducked = vad.ducked_frames
                    vad_muted = vad.muted_frames
                    phrase_mix = phrase_mix.overlay(harmony_seg)
                    harmony_applied = True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Harmony overlay failed on segment %d: %s", index, exc)

        phrase_file = phrases_dir / f"phrase_{index:02d}.mp3"
        phrase_mix.export(str(phrase_file), format="mp3")
        mixed_phrases.append(phrase_mix)
        lead_sequence.append(lead)

        lead_label = f"song_{lead}" if lead else "none"
        applied_overlays = list(overlay_stems) if overlay_from == "song_b" else []
        entry = PhraseScheduleEntry(
            index=index,
            lead=lead_label,
            start_sec=instr_section.start_sec,
            end_sec=instr_section.end_sec,
            duration_ms=target_ms,
            mashability_score=float(scores["mashability_score"]),
            harmonic_score=float(scores["harmonic_score"]),
            rhythmic_score=float(scores["rhythmic_score"]),
            spectral_score=float(scores["spectral_score"]),
            n_steps=int(scores["n_steps"]),
            cents=float(scores["cents"]),
            harmony=harmony_applied,
            harmony_suppressed_low_rhythm=harmony_suppressed,
            vad_overlap_frames=vad_overlap,
            vad_ducked_frames=vad_ducked,
            vad_muted_frames=vad_muted,
            instr_ducked_frames=instr_ducked_frames,
            enabled=True,
            phrase_file=str(phrase_file.name),
            section_name=segment.section_name,
            vocal_section_index=segment.vocal_section_index,
            instrumental_section_index=segment.instrumental_section_index,
            label=instr_section.label,
            high_pass_filter=apply_hpf,
            vocal_volume_db=vocal_gain,
            overlay_stems=applied_overlays,
            overlay_from=overlay_from if applied_overlays else "none",
        )
        schedule.append(entry)
        logger.info(
            "Segment %d — %s lead=%s score=%.3f duration_ms=%d muted=%d overlays=%s",
            index,
            segment.section_name,
            entry.lead,
            entry.mashability_score,
            target_ms,
            vad_muted,
            entry.overlay_stems,
        )

    if not mixed_phrases:
        raise RuntimeError("No sections were produced for dual-vocal mashup")

    fade_ms_list = _adaptive_fade_ms_list(lead_sequence)
    combined = crossfade_concatenate_adaptive(mixed_phrases, fade_ms_list)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(out), format="mp3")

    metadata: dict[str, Any] = {
        "creative_mode": creative_mode,
        "policy": policy,
        "first_lead": first_lead,
        "target_bpm": target_bpm,
        "bars_per_section": bars_per_section,
        "director_strict": not _allows_harmony(policy),
        "anchor": "song_a",
        "crossfade_ms": CROSSFADE_MS,
        "crossfade_adaptive_ms": fade_ms_list,
        "instr_duck_db": INSTR_DUCK_DB,
        "key_a": key_a,
        "key_b": key_b,
        "meeting_pc": meeting_pc,
        "shift_a": shift_a,
        "shift_b": shift_b,
        "shift_total": abs(shift_a) + abs(shift_b),
        "phrases": [asdict(p) for p in schedule],
        "sections_a": [s.to_prompt_dict() for s in sections_a],
        "sections_b": [s.to_prompt_dict() for s in sections_b],
    }
    if meter_a is not None:
        metadata["meter_a"] = meter_a
    if meter_b is not None:
        metadata["meter_b"] = meter_b
    if form_a is not None:
        metadata["form_a"] = form_a
    if form_b is not None:
        metadata["form_b"] = form_b
    if structure_meta_a is not None:
        metadata["structure_a"] = structure_meta_a
        metadata["structure_source_a"] = structure_meta_a.get("structure_source")
    if structure_meta_b is not None:
        metadata["structure_b"] = structure_meta_b
        metadata["structure_source_b"] = structure_meta_b.get("structure_source")
    # Convenience: primary source when both songs agree, else per-song keys above.
    src_a = (structure_meta_a or {}).get("structure_source")
    src_b = (structure_meta_b or {}).get("structure_source")
    if src_a and src_b:
        metadata["structure_source"] = src_a if src_a == src_b else f"{src_a}+{src_b}"
    elif src_a or src_b:
        metadata["structure_source"] = src_a or src_b
    if blueprint is not None:
        metadata["blueprint"] = blueprint.model_dump()
        metadata["arranging_reasoning"] = blueprint.arranging_reasoning
        if blueprint.actions:
            metadata["stem_actions"] = [a.model_dump() for a in blueprint.actions]
        if blueprint.song_b_hooks is not None:
            metadata["song_b_hooks"] = blueprint.song_b_hooks.model_dump()

    meta_path = session_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    session_mashup = session_dir / "mashup.mp3"
    session_mashup.write_bytes(out.read_bytes())

    return MashupResult(
        output_path=str(out),
        session_dir=str(session_dir),
        metadata=metadata,
    )


def np_clip_tempo(bpm: float) -> float:
    """Style-contrast helper: nudge BPM toward a danceable center within ±8%."""
    center = 120.0
    nudged = bpm + 0.5 * (center - bpm)
    return float(max(bpm * 0.92, min(bpm * 1.08, nudged)))


def build_bassline_mashup(
    *,
    bass_a: str,
    bass_b: str,
    drums_bed: str,
    other_bed: str | None,
    bpm_a: float,
    bpm_b: float,
    target_bpm: float,
    work_dir: Path,
    output_path: str,
    first_lead: Literal["a", "b"] = "a",
    max_sections: int = DEFAULT_MAX_SECTIONS,
    weights: MashabilityWeights | None = None,
    session_dir: Path | None = None,
    max_phrases: int | None = None,
) -> MashupResult:
    """Musician/bassline mode: alternate basslines over a drums(+other) bed."""
    if max_phrases is not None:
        max_sections = max_phrases

    work_dir.mkdir(parents=True, exist_ok=True)
    drums = AudioSegment.from_file(drums_bed)
    if other_bed:
        other = AudioSegment.from_file(other_bed)
        if other.frame_rate != drums.frame_rate:
            other = other.set_frame_rate(drums.frame_rate)
        if other.channels != drums.channels:
            other = other.set_channels(drums.channels)
        duration = max(len(drums), len(other))
        bed = (
            AudioSegment.silent(duration=duration, frame_rate=drums.frame_rate)
            .set_channels(drums.channels)
            .overlay(drums)
            .overlay(other)
        )
    else:
        bed = drums

    bed_path = work_dir / "bassline_bed.wav"
    bed.export(str(bed_path), format="wav")

    return build_dual_vocal_mashup(
        vocals_a=bass_a,
        vocals_b=bass_b,
        instrumental=str(bed_path),
        bpm_a=bpm_a,
        bpm_b=bpm_b,
        target_bpm=target_bpm,
        work_dir=work_dir / "bass_dual",
        output_path=output_path,
        policy="alternate",
        first_lead=first_lead,
        bars_per_section=DEFAULT_BARS_PER_SECTION,
        max_sections=max_sections,
        weights=weights,
        creative_mode="bassline",
        session_dir=session_dir,
    )


def reassemble_session(
    session_dir: str | Path,
    enabled_indices: list[int],
    output_path: str,
) -> str:
    """Concatenate enabled phrase MP3s from a mashup session (section editor)."""
    session = Path(session_dir)
    meta_path = session / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Session metadata missing: {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    phrases = metadata.get("phrases") or []
    enabled = set(int(i) for i in enabled_indices)

    segments: list[AudioSegment] = []
    for phrase in phrases:
        idx = int(phrase["index"])
        if idx not in enabled:
            phrase["enabled"] = False
            continue
        phrase["enabled"] = True
        fname = phrase.get("phrase_file") or f"phrase_{idx:02d}.mp3"
        path = session / "phrases" / fname
        if not path.is_file():
            raise FileNotFoundError(f"Missing phrase audio: {path}")
        segments.append(AudioSegment.from_file(path))

    if not segments:
        raise RuntimeError("No phrases enabled for reassembly")

    combined = crossfade_concatenate(segments, crossfade_ms=CROSSFADE_MS)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(out), format="mp3")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (session / "mashup.mp3").write_bytes(out.read_bytes())
    return str(out)
