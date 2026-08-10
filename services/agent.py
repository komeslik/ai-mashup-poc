"""LLM-driven mashup strategy via OpenAI or Gemini structured outputs."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, Field, model_validator

from services.audio import tempo_aware_stretch_rate
from services.structure import Section, is_high_energy_label, section_index_for_bar

logger = logging.getLogger(__name__)

SongSource = Literal["song_a", "song_b"]
VocalPick = Literal["song_a", "song_b", "none"]
ProviderName = Literal["openai", "gemini"]
PhraseVocalPolicy = Literal["alternate", "a_lead_b_harmony", "b_lead_a_harmony"]

SYSTEM_PROMPT = (
    "You are a DJ / music producer assistant. Given two songs' BPMs and an optional "
    "creative_mode, decide a mashup strategy for a dual-vocal mix. Both songs' vocals "
    "will be used across phrases; pick which song supplies the instrumental bed, which "
    "vocal leads first (vocal_source), target_bpm (usually the instrumental's BPM), and "
    "phrase_vocal_policy: 'alternate' (call-and-response, preferred), "
    "'a_lead_b_harmony', or 'b_lead_a_harmony'. "
    "vocal_source must differ from instrumental_source. "
    "For creative_mode=style_contrast, prefer bolder target_bpm choices still within "
    "a musically plausible range (avoid absurd stretches). "
    "stretch_factor MUST equal target_bpm / vocal_source_bpm (tempo-octave OK)."
)

ARRANGEMENT_SYSTEM_PROMPT = (
    "You are an expert audio arrangement engineer (AI DJ Director) for two-song mashups. "
    "You receive ROLE MAPS per 8-bar section with labels: intro, verse, buildup, chorus, "
    "drop, outro — plus energy, spectral_centroid, and vocal_density. "
    "Design a StemAction timeline snapped to 8-bar boundaries (bar_start % 8 == 0, "
    "bar_end > bar_start, both multiples of 8). "
    "STRICT RULES: "
    "(1) NEVER assign two vocal sources to the same bars — vocal_source is song_a OR "
    "song_b OR none (never both). "
    "(2) Align chorus/drop vocals over high-energy instrumental chorus/drop sections. "
    "(3) Prefer vocal_source=none on buildup bars so the instrumental can shine. "
    "(4) Feature BOTH singers across the mashup (section swap / call-and-response). "
    "(5) Keep 3–6 actions. "
    "(6) instrumental_source is constant for the whole mashup and must differ from the "
    "primary vocal_source. "
    "(7) target_bpm is usually the instrumental song BPM; stretch_factor MUST equal "
    "target_bpm / primary vocal BPM (tempo-octave OK). "
    "(8) high_pass_filter should be true for vocal beds; vocal_volume_db is usually 0. "
    "(9) Titles are creative context only — never invent bars that are not in the maps."
)


class MashupDecision(BaseModel):
    """Structured mixing strategy for a two-song mashup (policy-level)."""

    vocal_source: SongSource = Field(
        description="Which vocal leads first (alternate) or is primary lead (harmony policies)."
    )
    instrumental_source: SongSource = Field(
        description="Which song supplies the instrumental stem."
    )
    # Avoid Field(gt=0): Gemini's Schema rejects JSON Schema exclusiveMinimum.
    target_bpm: float = Field(description="BPM to align the mashup to.")
    stretch_factor: float = Field(
        description="Time-stretch factor for the preferred vocal: target_bpm / vocal_source_bpm.",
    )
    phrase_vocal_policy: PhraseVocalPolicy = Field(
        default="alternate",
        description=(
            "How to schedule both vocals: alternate leads per phrase, or one lead "
            "with the other as muted-on-overlap harmony."
        ),
    )

    @model_validator(mode="after")
    def sources_must_differ(self) -> MashupDecision:
        if self.vocal_source == self.instrumental_source:
            raise ValueError(
                "vocal_source and instrumental_source must come from different songs"
            )
        if self.target_bpm <= 0:
            raise ValueError("target_bpm must be positive")
        if self.stretch_factor <= 0:
            raise ValueError("stretch_factor must be positive")
        return self


class StemAction(BaseModel):
    """One 8-bar-snapped stem schedule block for the AI Director."""

    bar_start: int = Field(description="Inclusive start bar; multiple of 8.")
    bar_end: int = Field(description="Exclusive end bar; multiple of 8.")
    vocal_source: VocalPick = Field(
        description="Single vocal source for these bars, or none."
    )
    instrumental_source: SongSource = Field(
        description="Instrumental bed source (usually constant across actions)."
    )
    vocal_volume_db: float = Field(
        default=0.0,
        description="Vocal gain offset in dB.",
    )
    high_pass_filter: bool = Field(
        default=True,
        description="Apply ~100Hz low-cut to vocals.",
    )
    section_name: str = Field(
        default="",
        description="Short DJ cue name for this block.",
    )

    @model_validator(mode="after")
    def bars_must_snap(self) -> StemAction:
        if self.bar_end <= self.bar_start:
            raise ValueError("bar_end must be > bar_start")
        if self.bar_start % 8 != 0 or self.bar_end % 8 != 0:
            raise ValueError("bar_start and bar_end must be multiples of 8")
        return self


class ArrangementSegment(BaseModel):
    """One macro section in the mashup timeline (legacy executor shape)."""

    section_name: str = Field(
        description="Short DJ cue name, e.g. 'A chorus over B drop'."
    )
    vocal_source: VocalPick = Field(
        description="Which vocal to feature, or none for instrumental-only."
    )
    vocal_section_index: int = Field(
        description="Index into that vocal song's section map (ignored if vocal_source=none)."
    )
    instrumental_section_index: int = Field(
        description="Index into the instrumental song's section map."
    )
    harmony: bool = Field(
        default=False,
        description="If true, overlay the other vocal with hard VAD mute on overlap.",
    )
    vocal_volume_db: float = Field(default=0.0)
    high_pass_filter: bool = Field(default=True)


class ArrangementBlueprint(BaseModel):
    """LLM DJ Director bar-snapped stem plan (preferred schema)."""

    arrangement_reasoning: str = Field(
        description="Brief explanation of the arrangement arc."
    )
    instrumental_source: SongSource = Field(
        description="Which song supplies the instrumental bed for the whole mashup."
    )
    vocal_source: SongSource = Field(
        description="Which vocal leads first / is primary."
    )
    target_bpm: float = Field(description="BPM to align the mashup to.")
    stretch_factor: float = Field(
        description="target_bpm / vocal_source BPM (tempo-octave OK).",
    )
    phrase_vocal_policy: PhraseVocalPolicy = Field(
        default="alternate",
        description="Harmony only when UI selects a harmony policy; director default is alternate.",
    )
    actions: list[StemAction] = Field(
        description="Ordered 8-bar-snapped stem actions."
    )

    @model_validator(mode="after")
    def validate_arrangement(self) -> ArrangementBlueprint:
        if self.vocal_source == self.instrumental_source:
            raise ValueError(
                "vocal_source and instrumental_source must come from different songs"
            )
        if self.target_bpm <= 0:
            raise ValueError("target_bpm must be positive")
        if self.stretch_factor <= 0:
            raise ValueError("stretch_factor must be positive")
        if not self.actions:
            raise ValueError("actions must contain at least one StemAction")
        if len(self.actions) > 8:
            raise ValueError("actions too long; keep at most 8")
        # Enforce constant instrumental bed.
        for action in self.actions:
            if action.instrumental_source != self.instrumental_source:
                raise ValueError("all StemActions must share instrumental_source")
        return self


class MashupBlueprint(BaseModel):
    """Executable arrangement (section indices) used by the DSP mixer."""

    arranging_reasoning: str = Field(
        description="Brief explanation of the arrangement arc."
    )
    instrumental_source: SongSource = Field(
        description="Which song supplies the instrumental bed for the whole mashup."
    )
    vocal_source: SongSource = Field(
        description="Which vocal leads first / is primary."
    )
    target_bpm: float = Field(description="BPM to align the mashup to.")
    stretch_factor: float = Field(
        description="target_bpm / vocal_source BPM (tempo-octave OK).",
    )
    phrase_vocal_policy: PhraseVocalPolicy = Field(
        default="alternate",
        description="Harmony policy hint when segments request harmony.",
    )
    timeline: list[ArrangementSegment] = Field(
        description="Ordered macro sections to render."
    )
    actions: list[StemAction] = Field(
        default_factory=list,
        description="Optional original StemAction list for metadata.",
    )

    @model_validator(mode="after")
    def validate_blueprint(self) -> MashupBlueprint:
        if self.vocal_source == self.instrumental_source:
            raise ValueError(
                "vocal_source and instrumental_source must come from different songs"
            )
        if self.target_bpm <= 0:
            raise ValueError("target_bpm must be positive")
        if self.stretch_factor <= 0:
            raise ValueError("stretch_factor must be positive")
        if not self.timeline:
            raise ValueError("timeline must contain at least one segment")
        if len(self.timeline) > 8:
            raise ValueError("timeline too long; keep at most 8 segments")
        return self

    def as_decision(self) -> MashupDecision:
        return MashupDecision(
            vocal_source=self.vocal_source,
            instrumental_source=self.instrumental_source,
            target_bpm=self.target_bpm,
            stretch_factor=self.stretch_factor,
            phrase_vocal_policy=self.phrase_vocal_policy,
        )


def _fallback_decision(song_a_bpm: float, song_b_bpm: float) -> MashupDecision:
    """Deterministic POC fallback: B instrumental, alternate vocals starting with A."""
    target_bpm = float(song_b_bpm)
    stretch_factor = tempo_aware_stretch_rate(float(song_a_bpm), target_bpm)
    return MashupDecision(
        vocal_source="song_a",
        instrumental_source="song_b",
        target_bpm=target_bpm,
        stretch_factor=stretch_factor,
        phrase_vocal_policy="alternate",
    )


def _normalize_stretch(
    decision: MashupDecision,
    song_a_bpm: float,
    song_b_bpm: float,
) -> MashupDecision:
    """Recompute stretch_factor with tempo-octave handling for consistent DSP."""
    vocal_bpm = song_a_bpm if decision.vocal_source == "song_a" else song_b_bpm
    stretch_factor = tempo_aware_stretch_rate(vocal_bpm, decision.target_bpm)
    return decision.model_copy(update={"stretch_factor": stretch_factor})


def _normalize_blueprint_stretch(
    blueprint: MashupBlueprint,
    song_a_bpm: float,
    song_b_bpm: float,
) -> MashupBlueprint:
    vocal_bpm = song_a_bpm if blueprint.vocal_source == "song_a" else song_b_bpm
    stretch_factor = tempo_aware_stretch_rate(vocal_bpm, blueprint.target_bpm)
    return blueprint.model_copy(update={"stretch_factor": stretch_factor})


def _normalize_arrangement_stretch(
    arrangement: ArrangementBlueprint,
    song_a_bpm: float,
    song_b_bpm: float,
) -> ArrangementBlueprint:
    vocal_bpm = song_a_bpm if arrangement.vocal_source == "song_a" else song_b_bpm
    stretch_factor = tempo_aware_stretch_rate(vocal_bpm, arrangement.target_bpm)
    return arrangement.model_copy(update={"stretch_factor": stretch_factor})


def arrangement_to_mashup_blueprint(
    arrangement: ArrangementBlueprint,
    sections_a: list[Section],
    sections_b: list[Section],
    *,
    allow_harmony: bool = False,
) -> MashupBlueprint:
    """Convert bar-snapped StemActions into section-index timeline for the mixer."""
    instr_sections = (
        sections_a if arrangement.instrumental_source == "song_a" else sections_b
    )
    timeline: list[ArrangementSegment] = []
    for action in arrangement.actions:
        instr_idx = section_index_for_bar(instr_sections, action.bar_start)
        if action.vocal_source == "song_a":
            v_idx = section_index_for_bar(sections_a, action.bar_start)
        elif action.vocal_source == "song_b":
            v_idx = section_index_for_bar(sections_b, action.bar_start)
        else:
            v_idx = 0
        name = action.section_name or (
            f"{action.vocal_source} bars {action.bar_start}-{action.bar_end}"
        )
        timeline.append(
            ArrangementSegment(
                section_name=name,
                vocal_source=action.vocal_source,
                vocal_section_index=v_idx,
                instrumental_section_index=instr_idx,
                harmony=False,
                vocal_volume_db=action.vocal_volume_db,
                high_pass_filter=action.high_pass_filter,
            )
        )
    policy = arrangement.phrase_vocal_policy
    if not allow_harmony and policy in ("a_lead_b_harmony", "b_lead_a_harmony"):
        policy = "alternate"
    return MashupBlueprint(
        arranging_reasoning=arrangement.arrangement_reasoning,
        instrumental_source=arrangement.instrumental_source,
        vocal_source=arrangement.vocal_source,
        target_bpm=arrangement.target_bpm,
        stretch_factor=arrangement.stretch_factor,
        phrase_vocal_policy=policy,
        timeline=timeline,
        actions=list(arrangement.actions),
    )


def _clamp_blueprint_indices(
    blueprint: MashupBlueprint,
    sections_a: list[Section],
    sections_b: list[Section],
    *,
    allow_harmony: bool = False,
) -> MashupBlueprint:
    """Clamp section indices into available maps; strip harmony unless allowed."""
    n_a = max(len(sections_a), 1)
    n_b = max(len(sections_b), 1)
    instr_n = n_a if blueprint.instrumental_source == "song_a" else n_b
    fixed: list[ArrangementSegment] = []
    for seg in blueprint.timeline:
        if seg.vocal_source == "song_a":
            v_idx = int(seg.vocal_section_index) % n_a
        elif seg.vocal_source == "song_b":
            v_idx = int(seg.vocal_section_index) % n_b
        else:
            v_idx = 0
        i_idx = int(seg.instrumental_section_index) % instr_n
        harmony = bool(seg.harmony) and allow_harmony
        fixed.append(
            seg.model_copy(
                update={
                    "vocal_section_index": v_idx,
                    "instrumental_section_index": i_idx,
                    "harmony": harmony,
                }
            )
        )
    return blueprint.model_copy(update={"timeline": fixed})


def _sections_prompt_block(name: str, sections: list[Section]) -> str:
    rows = [s.to_prompt_dict() for s in sections]
    return f"{name} sections ({len(rows)}):\n{json.dumps(rows, indent=2)}"


def _user_prompt(
    song_a_bpm: float,
    song_b_bpm: float,
    creative_mode: str = "forced_match",
) -> str:
    return (
        f"Song A BPM: {song_a_bpm:.3f}\n"
        f"Song B BPM: {song_b_bpm:.3f}\n"
        f"creative_mode: {creative_mode}\n"
        "Return a structured mashup decision."
    )


def _arrangement_user_prompt(
    *,
    song_a_bpm: float,
    song_b_bpm: float,
    sections_a: list[Section],
    sections_b: list[Section],
    creative_mode: str = "forced_match",
    title_a: str | None = None,
    title_b: str | None = None,
) -> str:
    return (
        f"Song A title: {title_a or 'Song A'}\n"
        f"Song B title: {title_b or 'Song B'}\n"
        f"Song A BPM: {song_a_bpm:.3f}\n"
        f"Song B BPM: {song_b_bpm:.3f}\n"
        f"creative_mode: {creative_mode}\n"
        f"{_sections_prompt_block('Song A', sections_a)}\n"
        f"{_sections_prompt_block('Song B', sections_b)}\n"
        "Return an ArrangementBlueprint with StemAction bars snapped to multiples of 8."
    )


def _high_indices(sections: list[Section]) -> list[int]:
    highs = [s.index for s in sections if is_high_energy_label(s.label)]
    if highs:
        return highs
    if not sections:
        return [0]
    ranked = sorted(sections, key=lambda s: s.energy, reverse=True)
    return [ranked[0].index]


def _fallback_arrangement(
    song_a_bpm: float,
    song_b_bpm: float,
    sections_a: list[Section],
    sections_b: list[Section],
    *,
    creative_mode: str = "forced_match",
) -> ArrangementBlueprint:
    """Role-aware StemAction fallback: verse A, chorus A, chorus B, optional buildup mute."""
    del creative_mode
    decision = _fallback_decision(song_a_bpm, song_b_bpm)
    instr = decision.instrumental_source
    instr_sections = sections_a if instr == "song_a" else sections_b
    a_high = _high_indices(sections_a)
    b_high = _high_indices(sections_b)
    i_high = _high_indices(instr_sections)

    def _bars(section: Section) -> tuple[int, int]:
        return section.bar_start, section.bar_end

    actions: list[StemAction] = []
    # Intro / verse from A over early instrumental; mute if buildup.
    if sections_a and instr_sections:
        low_a = next(
            (
                s
                for s in sections_a
                if s.label in ("intro", "verse", "low", "buildup")
            ),
            sections_a[0],
        )
        early_i = instr_sections[0]
        b0, b1 = _bars(early_i)
        vocal: VocalPick = "none" if low_a.label == "buildup" else "song_a"
        actions.append(
            StemAction(
                bar_start=b0,
                bar_end=b1,
                vocal_source=vocal,
                instrumental_source=instr,
                section_name="Intro / verse (A)",
            )
        )

    # Prefer instrumental-only on buildup bars (director anti-collision / tension).
    buildup = next((s for s in instr_sections if s.label == "buildup"), None)
    if buildup:
        b0, b1 = _bars(buildup)
        if not actions or (b0, b1) != (actions[-1].bar_start, actions[-1].bar_end):
            actions.append(
                StemAction(
                    bar_start=b0,
                    bar_end=b1,
                    vocal_source="none",
                    instrumental_source=instr,
                    section_name="Buildup (instrumental)",
                )
            )

    a_sec = sections_a[a_high[0]] if sections_a else None
    i_sec = instr_sections[i_high[0]] if instr_sections else None
    if a_sec and i_sec:
        b0, b1 = _bars(i_sec)
        actions.append(
            StemAction(
                bar_start=b0,
                bar_end=b1,
                vocal_source="song_a",
                instrumental_source=instr,
                section_name="Chorus/Drop A",
            )
        )

    b_sec = sections_b[b_high[0]] if sections_b else None
    i_sec2 = instr_sections[i_high[min(1, len(i_high) - 1)]] if instr_sections else i_sec
    if b_sec and i_sec2:
        b0, b1 = _bars(i_sec2)
        # Avoid identical bar range as previous action.
        if not actions or (b0, b1) != (actions[-1].bar_start, actions[-1].bar_end):
            actions.append(
                StemAction(
                    bar_start=b0,
                    bar_end=b1,
                    vocal_source="song_b",
                    instrumental_source=instr,
                    section_name="Chorus/Drop B",
                )
            )
        else:
            # Shift to next instrumental section if available.
            alt = instr_sections[min(i_sec2.index + 1, len(instr_sections) - 1)]
            b0, b1 = _bars(alt)
            actions.append(
                StemAction(
                    bar_start=b0,
                    bar_end=b1,
                    vocal_source="song_b",
                    instrumental_source=instr,
                    section_name="Chorus/Drop B",
                )
            )

    if not actions:
        actions.append(
            StemAction(
                bar_start=0,
                bar_end=8,
                vocal_source="song_a",
                instrumental_source=instr,
                section_name="Fallback block",
            )
        )

    return ArrangementBlueprint(
        arrangement_reasoning=(
            "Fallback director: verse/intro (or mute on buildup), then A chorus/drop, "
            "then B chorus/drop over the instrumental bed — never dual lead vocals."
        ),
        instrumental_source=decision.instrumental_source,
        vocal_source=decision.vocal_source,
        target_bpm=decision.target_bpm,
        stretch_factor=decision.stretch_factor,
        phrase_vocal_policy="alternate",
        actions=actions,
    )


def _fallback_blueprint(
    song_a_bpm: float,
    song_b_bpm: float,
    sections_a: list[Section],
    sections_b: list[Section],
    *,
    creative_mode: str = "forced_match",
) -> MashupBlueprint:
    arrangement = _fallback_arrangement(
        song_a_bpm,
        song_b_bpm,
        sections_a,
        sections_b,
        creative_mode=creative_mode,
    )
    return arrangement_to_mashup_blueprint(
        arrangement, sections_a, sections_b, allow_harmony=False
    )


def _resolve_provider() -> ProviderName:
    raw = (os.getenv("LLM_PROVIDER") or "gemini").strip().lower()
    if raw not in ("openai", "gemini"):
        logger.warning("Unknown LLM_PROVIDER=%r; defaulting to gemini", raw)
        return "gemini"
    return raw  # type: ignore[return-value]


def _decide_with_openai(
    song_a_bpm: float,
    song_b_bpm: float,
    creative_mode: str = "forced_match",
) -> MashupDecision:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")

    client = OpenAI(api_key=api_key)
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _user_prompt(song_a_bpm, song_b_bpm, creative_mode),
            },
        ],
        response_format=MashupDecision,
        temperature=0.35 if creative_mode == "style_contrast" else 0.2,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("OpenAI returned an empty structured response")
    return _normalize_stretch(parsed, song_a_bpm, song_b_bpm)


def _decide_with_gemini(
    song_a_bpm: float,
    song_b_bpm: float,
    creative_mode: str = "forced_match",
) -> MashupDecision:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=_user_prompt(song_a_bpm, song_b_bpm, creative_mode),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.35 if creative_mode == "style_contrast" else 0.2,
            response_mime_type="application/json",
            response_schema=MashupDecision,
        ),
    )
    parsed = response.parsed
    if parsed is None:
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty structured response")
        parsed = MashupDecision.model_validate_json(text)
    elif not isinstance(parsed, MashupDecision):
        parsed = MashupDecision.model_validate(parsed)

    return _normalize_stretch(parsed, song_a_bpm, song_b_bpm)


def decide_mashup_strategy(
    song_a_bpm: float,
    song_b_bpm: float,
    *,
    creative_mode: str = "forced_match",
) -> MashupDecision:
    """
    Ask an LLM for a mashup strategy, falling back to a fixed rule on failure.

    Used for bassline mode; vocal mashups use decide_arrangement.
    """
    if song_a_bpm <= 0 or song_b_bpm <= 0:
        raise ValueError("BPM values must be positive")

    provider = _resolve_provider()
    logger.info("Mashup strategy provider: %s mode=%s", provider, creative_mode)

    try:
        if provider == "openai":
            return _decide_with_openai(song_a_bpm, song_b_bpm, creative_mode)
        return _decide_with_gemini(song_a_bpm, song_b_bpm, creative_mode)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "%s mashup decision failed (%s); using fallback",
            provider,
            exc,
        )
        return _fallback_decision(song_a_bpm, song_b_bpm)


def _arrangement_with_openai(
    *,
    song_a_bpm: float,
    song_b_bpm: float,
    sections_a: list[Section],
    sections_b: list[Section],
    creative_mode: str,
    title_a: str | None,
    title_b: str | None,
) -> ArrangementBlueprint:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")

    client = OpenAI(api_key=api_key)
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": ARRANGEMENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _arrangement_user_prompt(
                    song_a_bpm=song_a_bpm,
                    song_b_bpm=song_b_bpm,
                    sections_a=sections_a,
                    sections_b=sections_b,
                    creative_mode=creative_mode,
                    title_a=title_a,
                    title_b=title_b,
                ),
            },
        ],
        response_format=ArrangementBlueprint,
        temperature=0.25 if creative_mode == "style_contrast" else 0.1,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("OpenAI returned an empty arrangement blueprint")
    return parsed


def _arrangement_with_gemini(
    *,
    song_a_bpm: float,
    song_b_bpm: float,
    sections_a: list[Section],
    sections_b: list[Section],
    creative_mode: str,
    title_a: str | None,
    title_b: str | None,
) -> ArrangementBlueprint:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=_arrangement_user_prompt(
            song_a_bpm=song_a_bpm,
            song_b_bpm=song_b_bpm,
            sections_a=sections_a,
            sections_b=sections_b,
            creative_mode=creative_mode,
            title_a=title_a,
            title_b=title_b,
        ),
        config=types.GenerateContentConfig(
            system_instruction=ARRANGEMENT_SYSTEM_PROMPT,
            temperature=0.25 if creative_mode == "style_contrast" else 0.1,
            response_mime_type="application/json",
            response_schema=ArrangementBlueprint,
        ),
    )
    parsed = response.parsed
    if parsed is None:
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty arrangement blueprint")
        parsed = ArrangementBlueprint.model_validate_json(text)
    elif not isinstance(parsed, ArrangementBlueprint):
        parsed = ArrangementBlueprint.model_validate(parsed)
    return parsed


def decide_arrangement(
    song_a_bpm: float,
    song_b_bpm: float,
    sections_a: list[Section],
    sections_b: list[Section],
    *,
    creative_mode: str = "forced_match",
    title_a: str | None = None,
    title_b: str | None = None,
    allow_harmony: bool = False,
) -> MashupBlueprint:
    """
    Ask the LLM DJ Director for a StemAction arrangement, then convert to mixer timeline.

    Falls back to a deterministic role-aware StemAction plan on failure.
    """
    if song_a_bpm <= 0 or song_b_bpm <= 0:
        raise ValueError("BPM values must be positive")
    if not sections_a or not sections_b:
        raise ValueError("Both songs need at least one detected section")

    provider = _resolve_provider()
    logger.info(
        "Arrangement provider: %s mode=%s allow_harmony=%s",
        provider,
        creative_mode,
        allow_harmony,
    )

    try:
        if provider == "openai":
            arrangement = _arrangement_with_openai(
                song_a_bpm=song_a_bpm,
                song_b_bpm=song_b_bpm,
                sections_a=sections_a,
                sections_b=sections_b,
                creative_mode=creative_mode,
                title_a=title_a,
                title_b=title_b,
            )
        else:
            arrangement = _arrangement_with_gemini(
                song_a_bpm=song_a_bpm,
                song_b_bpm=song_b_bpm,
                sections_a=sections_a,
                sections_b=sections_b,
                creative_mode=creative_mode,
                title_a=title_a,
                title_b=title_b,
            )
        arrangement = _normalize_arrangement_stretch(
            arrangement, song_a_bpm, song_b_bpm
        )
        blueprint = arrangement_to_mashup_blueprint(
            arrangement,
            sections_a,
            sections_b,
            allow_harmony=allow_harmony,
        )
        return _clamp_blueprint_indices(
            blueprint, sections_a, sections_b, allow_harmony=allow_harmony
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "%s arrangement failed (%s); using fallback blueprint",
            provider,
            exc,
        )
        blueprint = _fallback_blueprint(
            song_a_bpm,
            song_b_bpm,
            sections_a,
            sections_b,
            creative_mode=creative_mode,
        )
        return _clamp_blueprint_indices(
            blueprint, sections_a, sections_b, allow_harmony=allow_harmony
        )


def blueprint_to_public_dict(blueprint: MashupBlueprint) -> dict[str, Any]:
    return blueprint.model_dump()
