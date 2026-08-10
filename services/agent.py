"""LLM-driven mashup strategy via OpenAI or Gemini structured outputs."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, Field, model_validator

from services.audio import tempo_aware_stretch_rate
from services.structure import Section, is_high_energy_label

logger = logging.getLogger(__name__)

SongSource = Literal["song_a", "song_b"]
VocalPick = Literal["song_a", "song_b", "none"]
OverlayStem = Literal["drums", "bass", "other"]
OverlayFrom = Literal["song_b", "none"]
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
    "Song A is ALWAYS the ANCHOR: its instrumental bed (drums+bass+other) is the spine "
    "of the entire mashup. Song B only contributes vocals and selective stem overlays "
    "(drums / bass / other) — never replace Song A's bed wholesale. "
    "You receive ROLE MAPS as form sections with labels: intro, verse, buildup, chorus, "
    "drop, bridge, outro — plus energy, spectral_centroid, vocal_density, and indices. "
    "First fill song_b_hooks describing what is musically interesting about Song B "
    "(e.g. darbuka in drums) and which overlay stems to prefer (default drums). "
    "Then design a StemAction timeline using SECTION INDICES: "
    "section_start inclusive, section_end exclusive on Song A's section map "
    "(section_end > section_start; indices must stay inside the map). "
    "STRICT RULES: "
    "(1) instrumental_source MUST be song_a on every action. "
    "(2) NEVER two lead vocals on the same sections — vocal_source is song_a OR song_b OR none. "
    "(3) Chronological Song A spine: first action MUST use Song A's intro (section 0) "
    "with vocal_source song_a (or none); last action MUST use Song A's outro "
    "(final section index) with vocal_source song_a (or none). "
    "(4) Middle: walk A section indices forward; feature Song B vocals / drums overlays "
    "on selected middle sections only. "
    "(5) Feature BOTH singers: at least one verse-like and one chorus-like block for each. "
    "(6) Prefer vocal_source=none on buildup sections. "
    "(7) Under active lead vocals (song_a or song_b), NEVER use overlay stem 'other' "
    "(Demucs residue bleed); drums only (bass rare). 'other' only when vocal_source=none. "
    "(8) overlay_from=song_b only for short sections; overlay_stems must be subset of "
    "song_b_hooks.preferred_overlay_stems; keep overlay_volume_db around -6. "
    "(9) Keep 4–7 actions. "
    "(10) target_bpm is Song A BPM; stretch_factor = target_bpm / Song A BPM (tempo-octave OK). "
    "(11) vocal_source primary field should be song_a (first featured singer). "
    "(12) Never invent section indices outside the maps; titles are creative context only."
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
    def validate_decision(self) -> MashupDecision:
        if self.target_bpm <= 0:
            raise ValueError("target_bpm must be positive")
        if self.stretch_factor <= 0:
            raise ValueError("stretch_factor must be positive")
        return self


class SongBHookPlan(BaseModel):
    """What is musically interesting to lift from Song B onto the Song A bed."""

    interesting_elements: str = Field(
        description="Short description of cool Song B elements (e.g. darbuka, flute)."
    )
    preferred_overlay_stems: list[OverlayStem] = Field(
        default_factory=lambda: ["drums"],
        description="Demucs stems from Song B worth overlaying (prefer drums).",
    )
    avoid: str = Field(
        default="",
        description="What not to lift from Song B (e.g. full bed, muddy bass).",
    )


class StemAction(BaseModel):
    """One section-index stem schedule block for the AI Director."""

    section_start: int = Field(
        description="Inclusive start section index on Song A's form map."
    )
    section_end: int = Field(
        description="Exclusive end section index on Song A's form map."
    )
    vocal_source: VocalPick = Field(
        description="Single vocal source for these sections, or none."
    )
    instrumental_source: SongSource = Field(
        default="song_a",
        description="Instrumental bed source — always song_a (anchor).",
    )
    overlay_stems: list[OverlayStem] = Field(
        default_factory=list,
        description="Selective Song B stems to layer on the A bed.",
    )
    overlay_from: OverlayFrom = Field(
        default="none",
        description="song_b to enable overlays; none for bed+vocals only.",
    )
    overlay_volume_db: float = Field(
        default=-6.0,
        description="Gain for Song B stem overlays.",
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
    def sections_must_be_valid(self) -> StemAction:
        if self.section_end <= self.section_start:
            raise ValueError("section_end must be > section_start")
        if self.section_start < 0:
            raise ValueError("section_start must be >= 0")
        if self.instrumental_source != "song_a":
            object.__setattr__(self, "instrumental_source", "song_a")
        stems = list(self.overlay_stems)
        if self.vocal_source != "none":
            stems = [s for s in stems if s != "other"]
        if self.overlay_from == "none":
            stems = []
        elif stems and self.overlay_from != "song_b":
            object.__setattr__(self, "overlay_from", "song_b")
        if self.overlay_from == "song_b" and not stems:
            object.__setattr__(self, "overlay_from", "none")
            stems = []
        object.__setattr__(self, "overlay_stems", stems)
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
    overlay_stems: list[OverlayStem] = Field(default_factory=list)
    overlay_from: OverlayFrom = Field(default="none")
    overlay_volume_db: float = Field(default=-6.0)


class ArrangementBlueprint(BaseModel):
    """LLM DJ Director section-index stem plan (preferred schema)."""

    arrangement_reasoning: str = Field(
        description="Brief explanation of the arrangement arc."
    )
    song_b_hooks: SongBHookPlan = Field(
        default_factory=lambda: SongBHookPlan(
            interesting_elements="unspecified",
            preferred_overlay_stems=["drums"],
            avoid="full instrumental bed",
        ),
        description="What to lift from Song B onto the Song A anchor bed.",
    )
    instrumental_source: SongSource = Field(
        default="song_a",
        description="Always song_a — the anchor bed.",
    )
    vocal_source: SongSource = Field(
        default="song_a",
        description="Which vocal leads first / is primary (usually song_a).",
    )
    target_bpm: float = Field(description="BPM to align the mashup to (Song A).")
    stretch_factor: float = Field(
        description="target_bpm / Song A BPM (tempo-octave OK).",
    )
    phrase_vocal_policy: PhraseVocalPolicy = Field(
        default="alternate",
        description="Harmony only when UI selects a harmony policy; director default is alternate.",
    )
    actions: list[StemAction] = Field(
        description="Ordered section-index stem actions."
    )

    @model_validator(mode="after")
    def validate_arrangement(self) -> ArrangementBlueprint:
        if self.target_bpm <= 0:
            raise ValueError("target_bpm must be positive")
        if self.stretch_factor <= 0:
            raise ValueError("stretch_factor must be positive")
        if not self.actions:
            raise ValueError("actions must contain at least one StemAction")
        if len(self.actions) > 48:
            raise ValueError("actions too long; keep at most 48")
        # Force Song A anchor bed on the blueprint and every action.
        object.__setattr__(self, "instrumental_source", "song_a")
        allowed = set(self.song_b_hooks.preferred_overlay_stems)
        fixed_actions: list[StemAction] = []
        for action in self.actions:
            stems = [s for s in action.overlay_stems if s in allowed] if allowed else []
            if not allowed:
                stems = list(action.overlay_stems)
            if action.vocal_source != "none":
                stems = [s for s in stems if s != "other"]
            fixed_actions.append(
                action.model_copy(
                    update={
                        "instrumental_source": "song_a",
                        "overlay_stems": stems if action.overlay_from == "song_b" else [],
                        "overlay_from": (
                            "song_b"
                            if action.overlay_from == "song_b" and stems
                            else "none"
                        ),
                    }
                )
            )
        object.__setattr__(self, "actions", fixed_actions)
        return self


class MashupBlueprint(BaseModel):
    """Executable arrangement (section indices) used by the DSP mixer."""

    arranging_reasoning: str = Field(
        description="Brief explanation of the arrangement arc."
    )
    instrumental_source: SongSource = Field(
        default="song_a",
        description="Always song_a — the anchor bed.",
    )
    vocal_source: SongSource = Field(
        default="song_a",
        description="Which vocal leads first / is primary.",
    )
    target_bpm: float = Field(description="BPM to align the mashup to.")
    stretch_factor: float = Field(
        description="target_bpm / Song A BPM (tempo-octave OK).",
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
    song_b_hooks: SongBHookPlan | None = Field(
        default=None,
        description="Hook plan from the director for metadata / UI.",
    )

    @model_validator(mode="after")
    def validate_blueprint(self) -> MashupBlueprint:
        if self.target_bpm <= 0:
            raise ValueError("target_bpm must be positive")
        if self.stretch_factor <= 0:
            raise ValueError("stretch_factor must be positive")
        if not self.timeline:
            raise ValueError("timeline must contain at least one segment")
        if len(self.timeline) > 48:
            raise ValueError("timeline too long; keep at most 48 segments")
        object.__setattr__(self, "instrumental_source", "song_a")
        return self

    def as_decision(self) -> MashupDecision:
        return MashupDecision(
            vocal_source=self.vocal_source,
            instrumental_source="song_a",
            target_bpm=self.target_bpm,
            stretch_factor=self.stretch_factor,
            phrase_vocal_policy=self.phrase_vocal_policy,
        )


def _fallback_decision(song_a_bpm: float, song_b_bpm: float) -> MashupDecision:
    """Deterministic POC fallback: Song A anchor bed, A vocals first."""
    del song_b_bpm
    target_bpm = float(song_a_bpm)
    stretch_factor = tempo_aware_stretch_rate(float(song_a_bpm), target_bpm)
    return MashupDecision(
        vocal_source="song_a",
        instrumental_source="song_a",
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
    del song_b_bpm
    # Anchor mashups stretch everything to Song A / target (usually A BPM).
    stretch_factor = tempo_aware_stretch_rate(song_a_bpm, blueprint.target_bpm)
    return blueprint.model_copy(update={"stretch_factor": stretch_factor})


def _normalize_arrangement_stretch(
    arrangement: ArrangementBlueprint,
    song_a_bpm: float,
    song_b_bpm: float,
) -> ArrangementBlueprint:
    del song_b_bpm
    target_bpm = float(song_a_bpm)
    stretch_factor = tempo_aware_stretch_rate(song_a_bpm, target_bpm)
    return arrangement.model_copy(
        update={
            "stretch_factor": stretch_factor,
            "instrumental_source": "song_a",
            "target_bpm": target_bpm,
        }
    )


def _strip_other_under_vocals(
    vocal_source: VocalPick,
    overlay_stems: list[OverlayStem],
    overlay_from: OverlayFrom,
) -> tuple[list[OverlayStem], OverlayFrom]:
    stems = list(overlay_stems)
    if vocal_source != "none":
        stems = [s for s in stems if s != "other"]
    if overlay_from != "song_b" or not stems:
        return [], "none"
    return stems, "song_b"


def arrangement_to_mashup_blueprint(
    arrangement: ArrangementBlueprint,
    sections_a: list[Section],
    sections_b: list[Section],
    *,
    allow_harmony: bool = False,
) -> MashupBlueprint:
    """Convert section-index StemActions into mixer timeline; inject A bookends."""
    n_a = max(len(sections_a), 1)
    n_b = max(len(sections_b), 1)
    last_a = n_a - 1
    actions = list(arrangement.actions)

    # Inject Song A intro / outro bookends when missing.
    if actions:
        first = actions[0]
        if first.section_start != 0 or first.vocal_source == "song_b":
            actions.insert(
                0,
                StemAction(
                    section_start=0,
                    section_end=1,
                    vocal_source="song_a",
                    instrumental_source="song_a",
                    section_name="A intro bookend",
                ),
            )
        last = actions[-1]
        if last.section_end < n_a or last.section_start > last_a or last.vocal_source == "song_b":
            # Append outro if last action doesn't cover / use A outro.
            covers_outro = last.section_start <= last_a < last.section_end
            if not covers_outro or last.vocal_source == "song_b":
                actions.append(
                    StemAction(
                        section_start=last_a,
                        section_end=n_a,
                        vocal_source="song_a",
                        instrumental_source="song_a",
                        section_name="A outro bookend",
                    )
                )

    timeline: list[ArrangementSegment] = []
    for action in actions:
        instr_idx = int(action.section_start) % n_a
        # Prefer the first index in the action range; clamp end exclusivity.
        if action.section_end > action.section_start + 1:
            # Multi-section blocks still pick the start section for the clip.
            instr_idx = max(0, min(n_a - 1, action.section_start))

        if action.vocal_source == "song_a":
            v_idx = instr_idx % n_a
        elif action.vocal_source == "song_b":
            v_idx = instr_idx % n_b
            instr_sec = pick_safe(sections_a, instr_idx)
            if instr_sec is not None and sections_b:
                match = next(
                    (s for s in sections_b if s.label == instr_sec.label),
                    None,
                )
                if match is not None:
                    v_idx = match.index
        else:
            v_idx = 0

        stems, overlay_from = _strip_other_under_vocals(
            action.vocal_source,
            list(action.overlay_stems),
            action.overlay_from,
        )
        name = action.section_name or (
            f"{action.vocal_source} sections {action.section_start}-{action.section_end}"
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
                overlay_stems=stems,
                overlay_from=overlay_from,
                overlay_volume_db=action.overlay_volume_db,
            )
        )
    policy = arrangement.phrase_vocal_policy
    if not allow_harmony and policy in ("a_lead_b_harmony", "b_lead_a_harmony"):
        policy = "alternate"
    return MashupBlueprint(
        arranging_reasoning=arrangement.arrangement_reasoning,
        instrumental_source="song_a",
        vocal_source=arrangement.vocal_source,
        target_bpm=arrangement.target_bpm,
        stretch_factor=arrangement.stretch_factor,
        phrase_vocal_policy=policy,
        timeline=timeline,
        actions=list(actions),
        song_b_hooks=arrangement.song_b_hooks,
    )


def pick_safe(sections: list[Section], index: int) -> Section | None:
    if not sections:
        return None
    return sections[int(index) % len(sections)]


def _clamp_blueprint_indices(
    blueprint: MashupBlueprint,
    sections_a: list[Section],
    sections_b: list[Section],
    *,
    allow_harmony: bool = False,
) -> MashupBlueprint:
    """Clamp section indices into available maps; strip harmony / other under leads."""
    n_a = max(len(sections_a), 1)
    n_b = max(len(sections_b), 1)
    instr_n = n_a  # Song A anchor
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
        stems, overlay_from = _strip_other_under_vocals(
            seg.vocal_source,
            list(seg.overlay_stems),
            seg.overlay_from,
        )
        fixed.append(
            seg.model_copy(
                update={
                    "vocal_section_index": v_idx,
                    "instrumental_section_index": i_idx,
                    "harmony": harmony,
                    "overlay_from": overlay_from,
                    "overlay_stems": stems,
                }
            )
        )
    return blueprint.model_copy(
        update={"timeline": fixed, "instrumental_source": "song_a"}
    )


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
        f"Song A title (ANCHOR): {title_a or 'Song A'}\n"
        f"Song B title (overlays + guest vocals): {title_b or 'Song B'}\n"
        f"Song A BPM: {song_a_bpm:.3f}\n"
        f"Song B BPM: {song_b_bpm:.3f}\n"
        f"creative_mode: {creative_mode}\n"
        "Song A is the fixed instrumental bed. "
        "Describe song_b_hooks first, then StemActions on Song A's SECTION indices "
        "(section_start inclusive, section_end exclusive). "
        "Cover Song A's FULL section spine in order (one action per A section index "
        "0..N-1) so mashup length stays close to Song A — do not skip middle sections "
        "or collapse to a short highlight reel. "
        "First action = A intro (index 0); last action = A outro (final index). "
        "On some mid sections you may use Song B vocals and/or drums overlays.\n"
        f"{_sections_prompt_block('Song A', sections_a)}\n"
        f"{_sections_prompt_block('Song B', sections_b)}\n"
        "Return an ArrangementBlueprint with instrumental_source=song_a."
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
    """
    Full Song A spine fallback: one StemAction per A section so mashup ≈ Song A length.

    Selective B vocals + drums overlay on a mid verse-like section when available.
    """
    del creative_mode
    decision = _fallback_decision(song_a_bpm, song_b_bpm)
    hooks = SongBHookPlan(
        interesting_elements="percussion / drums from Song B",
        preferred_overlay_stems=["drums"],
        avoid="replacing Song A bed wholesale; other under leads",
    )

    n_a = len(sections_a)
    last_idx = max(n_a - 1, 0)
    # Pick one mid section for a B guest vocal + drums overlay.
    guest_idx: int | None = None
    if n_a >= 4 and sections_b:
        mid_candidates = [
            s.index
            for s in sections_a[1:last_idx]
            if s.label in ("verse", "bridge", "other", "buildup")
        ]
        if not mid_candidates:
            mid_candidates = [s.index for s in sections_a[1:last_idx]]
        if mid_candidates:
            guest_idx = mid_candidates[len(mid_candidates) // 2]

    actions: list[StemAction] = []
    if not sections_a:
        actions.append(
            StemAction(
                section_start=0,
                section_end=1,
                vocal_source="song_a",
                instrumental_source="song_a",
                section_name="Fallback A block",
            )
        )
    else:
        for sec in sections_a:
            idx = sec.index
            if guest_idx is not None and idx == guest_idx:
                actions.append(
                    StemAction(
                        section_start=idx,
                        section_end=idx + 1,
                        vocal_source="song_b",
                        instrumental_source="song_a",
                        overlay_from="song_b",
                        overlay_stems=["drums"],
                        overlay_volume_db=-6.0,
                        section_name=f"B guest over A {sec.label} ({idx})",
                    )
                )
            elif sec.label == "buildup" and idx not in (0, last_idx):
                actions.append(
                    StemAction(
                        section_start=idx,
                        section_end=idx + 1,
                        vocal_source="none",
                        instrumental_source="song_a",
                        section_name=f"A buildup ({idx})",
                    )
                )
            else:
                actions.append(
                    StemAction(
                        section_start=idx,
                        section_end=idx + 1,
                        vocal_source="song_a",
                        instrumental_source="song_a",
                        section_name=f"A {sec.label or 'section'} ({idx})",
                    )
                )

    return ArrangementBlueprint(
        arrangement_reasoning=(
            "Fallback full Song A spine: one action per A section so mashup length "
            "tracks Song A; optional mid B guest vocal + drums overlay."
        ),
        song_b_hooks=hooks,
        instrumental_source="song_a",
        vocal_source="song_a",
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
