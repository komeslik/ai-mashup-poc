"""LLM-driven mashup strategy via OpenAI or Gemini structured outputs."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, Field, model_validator

from services.audio import tempo_aware_stretch_rate
from services.structure import Section

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
    "You are a creative DJ director for two-song mashups. You receive timed SECTION MAPS "
    "with real start/end seconds and heuristic labels (intro/low/high/outro). "
    "high ≈ chorus/drop energy; low ≈ verse/build. "
    "Design a macro-structure arrangement: full sections (8 bars), NOT word-sized chops. "
    "Prefer arcs like verse→build→chorus, and feature BOTH singers across the timeline "
    "(e.g. Song A high/chorus, then Song B high/chorus). "
    "Rules: "
    "(1) Only reference section indices that exist in the provided maps — never invent times. "
    "(2) instrumental_source is constant for the whole mashup and must differ from at least "
    "one vocal_source used in the timeline. "
    "(3) vocal_section_index indexes the chosen vocal song's map; "
    "instrumental_section_index indexes the instrumental song's map. "
    "(4) Prefer pairing high vocal sections over high instrumental sections when possible. "
    "(5) Keep timeline length between 3 and 6 segments. "
    "(6) Titles are creative context only — do not invent timestamps from titles. "
    "(7) target_bpm is usually the instrumental song's BPM; stretch_factor MUST equal "
    "target_bpm / first-lead vocal BPM (tempo-octave OK). "
    "(8) vocal_source on MashupBlueprint is which singer leads the first segment / primary lead."
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
            "with the other as ducked harmony."
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


class ArrangementSegment(BaseModel):
    """One macro section in the mashup timeline."""

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
        description="If true, duck-overlay the other vocal briefly.",
    )


class MashupBlueprint(BaseModel):
    """LLM DJ Director arrangement executed by the DSP mixer."""

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


def _clamp_blueprint_indices(
    blueprint: MashupBlueprint,
    sections_a: list[Section],
    sections_b: list[Section],
) -> MashupBlueprint:
    """Clamp section indices into available maps; drop empty-safe timeline."""
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
        fixed.append(
            seg.model_copy(
                update={
                    "vocal_section_index": v_idx,
                    "instrumental_section_index": i_idx,
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
        "Return a MashupBlueprint with a macro-structure timeline."
    )


def _high_indices(sections: list[Section]) -> list[int]:
    highs = [s.index for s in sections if s.label == "high"]
    if highs:
        return highs
    if not sections:
        return [0]
    ranked = sorted(sections, key=lambda s: s.energy, reverse=True)
    return [ranked[0].index]


def _fallback_blueprint(
    song_a_bpm: float,
    song_b_bpm: float,
    sections_a: list[Section],
    sections_b: list[Section],
    *,
    creative_mode: str = "forced_match",
) -> MashupBlueprint:
    """Alternate highest-energy sections from each song over B's bed by default."""
    decision = _fallback_decision(song_a_bpm, song_b_bpm)
    instr = decision.instrumental_source
    instr_sections = sections_a if instr == "song_a" else sections_b
    a_high = _high_indices(sections_a)
    b_high = _high_indices(sections_b)
    i_high = _high_indices(instr_sections)

    timeline: list[ArrangementSegment] = []
    # Intro-ish: low/intro from A over early instrumental if available.
    if sections_a and instr_sections:
        low_a = next((s for s in sections_a if s.label in ("intro", "low")), sections_a[0])
        early_i = instr_sections[0]
        timeline.append(
            ArrangementSegment(
                section_name="Intro / verse (A)",
                vocal_source="song_a",
                vocal_section_index=low_a.index,
                instrumental_section_index=early_i.index,
                harmony=False,
            )
        )
    # Main: A high then B high over instrumental highs.
    timeline.append(
        ArrangementSegment(
            section_name="Chorus A",
            vocal_source="song_a",
            vocal_section_index=a_high[0],
            instrumental_section_index=i_high[0],
            harmony=False,
        )
    )
    timeline.append(
        ArrangementSegment(
            section_name="Chorus B",
            vocal_source="song_b",
            vocal_section_index=b_high[0],
            instrumental_section_index=i_high[min(1, len(i_high) - 1)],
            harmony=creative_mode == "style_contrast",
        )
    )
    if len(a_high) > 1:
        timeline.append(
            ArrangementSegment(
                section_name="Chorus A reprise",
                vocal_source="song_a",
                vocal_section_index=a_high[min(1, len(a_high) - 1)],
                instrumental_section_index=i_high[0],
                harmony=False,
            )
        )

    return MashupBlueprint(
        arranging_reasoning=(
            "Fallback: verse/low from A, then alternate high/chorus sections from A and B "
            "over the instrumental bed."
        ),
        instrumental_source=decision.instrumental_source,
        vocal_source=decision.vocal_source,
        target_bpm=decision.target_bpm,
        stretch_factor=decision.stretch_factor,
        phrase_vocal_policy=decision.phrase_vocal_policy,
        timeline=timeline,
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


def _arrangement_with_openai(
    *,
    song_a_bpm: float,
    song_b_bpm: float,
    sections_a: list[Section],
    sections_b: list[Section],
    creative_mode: str,
    title_a: str | None,
    title_b: str | None,
) -> MashupBlueprint:
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
        response_format=MashupBlueprint,
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
) -> MashupBlueprint:
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
            response_schema=MashupBlueprint,
        ),
    )
    parsed = response.parsed
    if parsed is None:
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty arrangement blueprint")
        parsed = MashupBlueprint.model_validate_json(text)
    elif not isinstance(parsed, MashupBlueprint):
        parsed = MashupBlueprint.model_validate(parsed)
    return parsed


def decide_mashup_strategy(
    song_a_bpm: float,
    song_b_bpm: float,
    *,
    creative_mode: str = "forced_match",
) -> MashupDecision:
    """
    Ask an LLM for a mashup strategy, falling back to a fixed rule on failure.

    Provider is selected with ``LLM_PROVIDER`` (``openai`` or ``gemini``).
    Both vocals are scheduled across phrases; the decision picks instrumental bed,
    first/primary lead, target BPM, and dual-vocal policy.
    """
    if song_a_bpm <= 0 or song_b_bpm <= 0:
        raise ValueError("BPM values must be positive")

    provider = _resolve_provider()
    logger.info("Mashup strategy provider: %s mode=%s", provider, creative_mode)

    try:
        if provider == "openai":
            return _decide_with_openai(song_a_bpm, song_b_bpm, creative_mode)
        return _decide_with_gemini(song_a_bpm, song_b_bpm, creative_mode)
    except Exception as exc:  # noqa: BLE001 — keep POC resilient overnight
        logger.warning(
            "%s mashup decision failed (%s); using fallback",
            provider,
            exc,
        )
        return _fallback_decision(song_a_bpm, song_b_bpm)


def decide_arrangement(
    song_a_bpm: float,
    song_b_bpm: float,
    sections_a: list[Section],
    sections_b: list[Section],
    *,
    creative_mode: str = "forced_match",
    title_a: str | None = None,
    title_b: str | None = None,
) -> MashupBlueprint:
    """
    Ask the LLM DJ Director for a section-level arrangement blueprint.

    Falls back to a deterministic high-energy alternate timeline on failure.
    """
    if song_a_bpm <= 0 or song_b_bpm <= 0:
        raise ValueError("BPM values must be positive")
    if not sections_a or not sections_b:
        raise ValueError("Both songs need at least one detected section")

    provider = _resolve_provider()
    logger.info("Arrangement provider: %s mode=%s", provider, creative_mode)

    try:
        if provider == "openai":
            blueprint = _arrangement_with_openai(
                song_a_bpm=song_a_bpm,
                song_b_bpm=song_b_bpm,
                sections_a=sections_a,
                sections_b=sections_b,
                creative_mode=creative_mode,
                title_a=title_a,
                title_b=title_b,
            )
        else:
            blueprint = _arrangement_with_gemini(
                song_a_bpm=song_a_bpm,
                song_b_bpm=song_b_bpm,
                sections_a=sections_a,
                sections_b=sections_b,
                creative_mode=creative_mode,
                title_a=title_a,
                title_b=title_b,
            )
        blueprint = _normalize_blueprint_stretch(blueprint, song_a_bpm, song_b_bpm)
        return _clamp_blueprint_indices(blueprint, sections_a, sections_b)
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
        return _clamp_blueprint_indices(blueprint, sections_a, sections_b)


def blueprint_to_public_dict(blueprint: MashupBlueprint) -> dict[str, Any]:
    return blueprint.model_dump()
