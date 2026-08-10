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
from services.structure import Section, detect_sections, pick_section
from services.vad import apply_overlap_duck

logger = logging.getLogger(__name__)

PhraseVocalPolicy = Literal["alternate", "a_lead_b_harmony", "b_lead_a_harmony"]
CreativeMode = Literal["forced_match", "style_contrast", "bassline"]

HARMONY_GAIN_DB = -7.0
STRETCH_EPSILON = 0.01
VOCAL_HPF_HZ = 100.0
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
    enabled: bool = True
    phrase_file: str | None = None
    section_name: str | None = None
    vocal_section_index: int | None = None
    instrumental_section_index: int | None = None
    label: str | None = None


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
) -> tuple[AudioSegment, dict[str, float | int | bool]]:
    """
    Slice a contiguous vocal section; score/key-match against the instrumental section.

    Never searches the whole song for a different vocal window.
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
        )
        n_steps = int(alignment.n_steps)
        scores: dict[str, float | int | bool] = {
            "mashability_score": float(alignment.score),
            "harmonic_score": float(alignment.harmonic_score),
            "rhythmic_score": float(alignment.rhythmic_score),
            "spectral_score": float(alignment.spectral_score),
            "n_steps": n_steps,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Section mashability failed (%s); using key match", exc)
        n_steps = _key_steps(vocal_clip, instrumental_phrase_path)
        scores = {
            "mashability_score": 0.0,
            "harmonic_score": 0.0,
            "rhythmic_score": 1.0,
            "spectral_score": 0.0,
            "n_steps": n_steps,
        }

    logger.info(
        "Section vocal %s — contiguous [%.1f–%.1f] score=%.3f n_steps=%+d",
        tag,
        vocal_section.start_sec,
        vocal_section.end_sec,
        scores["mashability_score"],
        scores["n_steps"],
    )
    cents = 0.0
    pitched, cents = pitch_shift_with_cents(
        vocal_clip,
        str(work_dir / f"{tag}_pitched.wav"),
        float(scores["n_steps"]),
        apply_tuning_correction=True,
    )
    scores["cents"] = float(cents)
    return AudioSegment.from_file(pitched), scores


def _lead_letter(vocal_source: str) -> Literal["a", "b"] | None:
    if vocal_source == "song_a":
        return "a"
    if vocal_source == "song_b":
        return "b"
    return None


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
    max_sections: int = DEFAULT_MAX_SECTIONS,
    weights: MashabilityWeights | None = None,
    creative_mode: CreativeMode = "forced_match",
    session_dir: Path | None = None,
    blueprint: MashupBlueprint | None = None,
    sections_a: list[Section] | None = None,
    sections_b: list[Section] | None = None,
    # Back-compat aliases from older call sites.
    beats_per_phrase: int = 8,
    bars_per_phrase: int | None = None,
    max_phrases: int | None = None,
) -> MashupResult:
    """
    Build a mashup from contiguous section clips (macro structure).

    When *blueprint* is provided, its timeline drives vocal/instrumental section picks.
    Otherwise sections are detected on the instrumental bed and leads alternate.
    """
    del beats_per_phrase  # unused; kept for call-site compatibility
    if bars_per_phrase is not None:
        bars_per_section = bars_per_phrase
    if max_phrases is not None:
        max_sections = max_phrases

    work_dir.mkdir(parents=True, exist_ok=True)
    if session_dir is None:
        session_dir = work_dir / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    phrases_dir = session_dir / "phrases"
    phrases_dir.mkdir(exist_ok=True)

    stretch_bpm = target_bpm
    if creative_mode == "style_contrast":
        stretch_bpm = float(np_clip_tempo(target_bpm))

    stretched_a = _stretch_to_bpm(
        vocals_a,
        str(work_dir / "vocals_a_stretched.wav"),
        bpm_a,
        stretch_bpm,
    )
    stretched_b = _stretch_to_bpm(
        vocals_b,
        str(work_dir / "vocals_b_stretched.wav"),
        bpm_b,
        stretch_bpm,
    )

    if sections_a is None:
        sections_a = detect_sections(
            vocals_a, bars_per_section=bars_per_section, max_sections=max_sections
        )
    if sections_b is None:
        sections_b = detect_sections(
            vocals_b, bars_per_section=bars_per_section, max_sections=max_sections
        )

    # Instrumental sections: prefer map from the instrumental-source song when
    # blueprint is present; otherwise detect on the bed itself.
    if blueprint is not None:
        if blueprint.instrumental_source == "song_a":
            sections_instr = list(sections_a)
        else:
            sections_instr = list(sections_b)
    else:
        sections_instr = detect_sections(
            instrumental,
            bars_per_section=bars_per_section,
            max_sections=max_sections,
        )

    sections_a_stretched = _scale_sections_after_stretch(sections_a, bpm_a, stretch_bpm)
    sections_b_stretched = _scale_sections_after_stretch(sections_b, bpm_b, stretch_bpm)

    if blueprint is not None:
        timeline = list(blueprint.timeline)
    else:
        timeline = _heuristic_timeline(
            sections_instr, sections_a, sections_b, policy, first_lead
        )

    logger.info(
        "Section mashup: %d segments, policy=%s, mode=%s, blueprint=%s",
        len(timeline),
        policy,
        creative_mode,
        blueprint is not None,
    )

    mixed_phrases: list[AudioSegment] = []
    schedule: list[PhraseScheduleEntry] = []

    for index, segment in enumerate(timeline):
        instr_section = pick_section(sections_instr, segment.instrumental_section_index)
        instr_phrase = extract_audio_segment(
            instrumental,
            str(work_dir / f"instr_section_{index}.wav"),
            instr_section.start_sec,
            instr_section.end_sec,
        )
        instrumental_seg = AudioSegment.from_file(instr_phrase)
        target_ms = len(instrumental_seg)

        lead = _lead_letter(segment.vocal_source)
        harmony = bool(segment.harmony) or (
            lead is not None and _allows_harmony(policy)
        )

        canvas = AudioSegment.silent(
            duration=target_ms,
            frame_rate=instrumental_seg.frame_rate,
        ).set_channels(instrumental_seg.channels)
        phrase_mix = canvas.overlay(instrumental_seg)

        scores: dict[str, float | int | bool] = {
            "mashability_score": 0.0,
            "harmonic_score": 0.0,
            "rhythmic_score": 1.0,
            "spectral_score": 0.0,
            "n_steps": 0,
            "cents": 0.0,
        }
        lead_seg: AudioSegment | None = None
        vocal_section_used: Section | None = None

        if lead is not None:
            if lead == "a":
                lead_path = stretched_a
                other_path = stretched_b
                vocal_map = sections_a_stretched
                other_map = sections_b_stretched
            else:
                lead_path = stretched_b
                other_path = stretched_a
                vocal_map = sections_b_stretched
                other_map = sections_a_stretched

            vocal_section_used = pick_section(vocal_map, segment.vocal_section_index)
            lead_seg, scores = _prepare_contiguous_lead(
                lead_path,
                vocal_section_used,
                instr_phrase,
                work_dir,
                tag=f"s{index}_lead_{lead}",
                weights=weights,
            )
            lead_seg = fit_length(lead_seg, target_ms)
            lead_seg = _match_format(lead_seg, instrumental_seg)
            try:
                lead_seg = highpass_segment(lead_seg, cutoff_hz=VOCAL_HPF_HZ)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Lead HPF failed: %s", exc)
            lead_seg = match_loudness(lead_seg, instrumental_seg)
            phrase_mix = phrase_mix.overlay(lead_seg)

        harmony_applied = False
        harmony_suppressed = False
        vad_overlap = 0
        vad_ducked = 0

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
                    # Prefer a high other-vocal section for harmony color.
                    high_other = next(
                        (s for s in other_map if s.label == "high"), other_section
                    )
                    harmony_seg, _ = _prepare_contiguous_lead(
                        other_path,
                        high_other,
                        instr_phrase,
                        work_dir,
                        tag=f"s{index}_harm",
                        weights=weights,
                    )
                    harmony_seg = fit_length(harmony_seg, target_ms)
                    harmony_seg = _match_format(harmony_seg, instrumental_seg)
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
                    vad = apply_overlap_duck(lead_seg, harmony_seg, duck_db=-18.0)
                    harmony_seg = vad.segment
                    vad_overlap = vad.overlap_frames
                    vad_ducked = vad.ducked_frames
                    phrase_mix = phrase_mix.overlay(harmony_seg)
                    harmony_applied = True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Harmony overlay failed on segment %d: %s", index, exc)

        phrase_file = phrases_dir / f"phrase_{index:02d}.mp3"
        phrase_mix.export(str(phrase_file), format="mp3")
        mixed_phrases.append(phrase_mix)

        lead_label = f"song_{lead}" if lead else "none"
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
            enabled=True,
            phrase_file=str(phrase_file.name),
            section_name=segment.section_name,
            vocal_section_index=segment.vocal_section_index,
            instrumental_section_index=segment.instrumental_section_index,
            label=instr_section.label,
        )
        schedule.append(entry)
        logger.info(
            "Segment %d — %s lead=%s score=%.3f duration_ms=%d",
            index,
            segment.section_name,
            entry.lead,
            entry.mashability_score,
            target_ms,
        )

    if not mixed_phrases:
        raise RuntimeError("No sections were produced for dual-vocal mashup")

    combined = sum(mixed_phrases[1:], mixed_phrases[0])
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(out), format="mp3")

    metadata: dict[str, Any] = {
        "creative_mode": creative_mode,
        "policy": policy,
        "first_lead": first_lead,
        "target_bpm": target_bpm,
        "bars_per_section": bars_per_section,
        "phrases": [asdict(p) for p in schedule],
        "sections_a": [s.to_prompt_dict() for s in sections_a],
        "sections_b": [s.to_prompt_dict() for s in sections_b],
    }
    if blueprint is not None:
        metadata["blueprint"] = blueprint.model_dump()
        metadata["arranging_reasoning"] = blueprint.arranging_reasoning

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

    combined = sum(segments[1:], segments[0])
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(out), format="mp3")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (session / "mashup.mp3").write_bytes(out.read_bytes())
    return str(out)
