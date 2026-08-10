"""LLM song-form analysis + DSP helpers (allin1 lives in allin1_structure)."""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from services.structure import (
    Section,
    detect_sections,
    normalize_section_label,
    sections_from_form_specs,
    snap_form_timestamps,
)

logger = logging.getLogger(__name__)

SectionLabelName = Literal[
    "intro",
    "verse",
    "buildup",
    "chorus",
    "drop",
    "bridge",
    "outro",
    "other",
]


class FormSectionSpec(BaseModel):
    """One timed form section guessed by the LLM."""

    name: str = Field(description="Human-readable section name, e.g. Verse 1.")
    label: SectionLabelName = Field(
        description="Normalized role label for arrangement."
    )
    start_sec: float = Field(description="Section start in seconds.")
    end_sec: float = Field(description="Section end in seconds.")
    approx_beats: int = Field(default=0, description="Approximate beat count.")
    description: str = Field(default="", description="Short musical note.")


class SongFormAnalysis(BaseModel):
    """Structured song form from title + duration (clocks are approximate)."""

    title: str
    bpm: float = Field(description="Estimated or provided BPM.")
    time_signature_numerator: int = Field(default=4, ge=2, le=12)
    time_signature_denominator: int = Field(default=4)
    style_notes: str = Field(default="", description="Brief style / genre note.")
    total_duration_sec: float = Field(
        description="Expected full-song duration the timestamps cover."
    )
    sections: list[FormSectionSpec] = Field(
        description="Ordered contiguous-ish sections covering the song."
    )

    @model_validator(mode="after")
    def _normalize(self) -> SongFormAnalysis:
        if self.bpm <= 0:
            raise ValueError("bpm must be positive")
        if self.total_duration_sec <= 0:
            raise ValueError("total_duration_sec must be positive")
        if not self.sections:
            raise ValueError("sections must be non-empty")
        cleaned: list[FormSectionSpec] = []
        for sec in self.sections:
            label = normalize_section_label(sec.label)  # type: ignore[arg-type]
            cleaned.append(
                sec.model_copy(
                    update={
                        "label": label if label in (
                            "intro",
                            "verse",
                            "buildup",
                            "chorus",
                            "drop",
                            "bridge",
                            "outro",
                            "other",
                        )
                        else "other"
                    }
                )
            )
        object.__setattr__(self, "sections", cleaned)
        return self

    def to_form_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "bpm": self.bpm,
            "time_signature_numerator": self.time_signature_numerator,
            "time_signature_denominator": self.time_signature_denominator,
            "style_notes": self.style_notes,
            "total_duration_sec": self.total_duration_sec,
            "sections": [s.model_dump() for s in self.sections],
        }


FORM_SYSTEM_PROMPT = (
    "You analyze popular-song form from title metadata and duration only. "
    "Return a JSON SongFormAnalysis: intro → verses/choruses/bridges → outro. "
    "Timestamps are guesses that must span 0..total_duration_sec without large gaps. "
    "Prefer 6–14 sections. Labels must be one of: intro, verse, buildup, chorus, "
    "drop, bridge, outro, other. Do not invent silence; cover the whole duration."
)


def _resolve_provider() -> Literal["openai", "gemini"]:
    raw = (os.getenv("LLM_PROVIDER") or "gemini").strip().lower()
    if raw not in ("openai", "gemini"):
        return "gemini"
    return raw  # type: ignore[return-value]


def _form_user_prompt(title: str, bpm: float, duration_sec: float) -> str:
    return (
        f"Song title: {title}\n"
        f"Measured/estimated BPM: {bpm:.3f}\n"
        f"Measured duration seconds: {duration_sec:.3f}\n"
        "Propose a full form with start_sec/end_sec covering the duration."
    )


def analyze_song_form(
    title: str,
    bpm: float,
    duration_sec: float,
) -> SongFormAnalysis:
    """Ask the configured LLM for a SongFormAnalysis."""
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    provider = _resolve_provider()
    logger.info("Song form provider=%s title=%r", provider, title)
    if provider == "openai":
        return _analyze_form_openai(title, bpm, duration_sec)
    return _analyze_form_gemini(title, bpm, duration_sec)


def _analyze_form_openai(title: str, bpm: float, duration_sec: float) -> SongFormAnalysis:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    client = OpenAI(api_key=api_key)
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": FORM_SYSTEM_PROMPT},
            {"role": "user", "content": _form_user_prompt(title, bpm, duration_sec)},
        ],
        response_format=SongFormAnalysis,
        temperature=0.2,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("OpenAI returned empty SongFormAnalysis")
    return parsed


def _analyze_form_gemini(title: str, bpm: float, duration_sec: float) -> SongFormAnalysis:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=_form_user_prompt(title, bpm, duration_sec),
        config=types.GenerateContentConfig(
            system_instruction=FORM_SYSTEM_PROMPT,
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=SongFormAnalysis,
        ),
    )
    parsed = response.parsed
    if parsed is None:
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned empty SongFormAnalysis")
        return SongFormAnalysis.model_validate_json(text)
    if isinstance(parsed, SongFormAnalysis):
        return parsed
    return SongFormAnalysis.model_validate(parsed)


def _measured_duration(file_path: str) -> float:
    import librosa

    return float(librosa.get_duration(path=file_path))


def resolve_sections_llm(
    file_path: str,
    title: str,
    bpm: float,
    vocals_path: str | None = None,
    *,
    measured_duration_sec: float | None = None,
) -> tuple[list[Section], dict[str, Any] | None, dict[str, Any]]:
    """
    LLM form timestamps snapped onto the file duration, then Section features.

    Returns (sections, form_dict, metadata).
    """
    measured = float(measured_duration_sec or _measured_duration(file_path))
    try:
        analysis = analyze_song_form(title, bpm, measured)
        specs = snap_form_timestamps(
            [s.model_dump() for s in analysis.sections],
            measured,
            expected_duration_sec=analysis.total_duration_sec,
        )
        meter = int(analysis.time_signature_numerator or 4)
        sections = sections_from_form_specs(
            file_path,
            specs,
            bpm=bpm,
            vocals_path=vocals_path,
            meter_numerator=meter,
        )
        if not sections:
            raise RuntimeError("LLM form produced no sections")
        form_dict = analysis.to_form_dict()
        form_dict["total_duration_sec"] = measured
        form_dict["sections"] = [
            {
                "name": s.name or s.label,
                "label": s.label,
                "start_sec": s.start_sec,
                "end_sec": s.end_sec,
                "approx_beats": s.approx_beats,
                "description": s.description,
            }
            for s in sections
        ]
        meta = {
            "structure_source": "llm",
            "structure_mode": "llm",
            "bpm": bpm,
            "meter_numerator": meter,
            "meter_denominator": 4,
            "title": title,
            "measured_bpm": bpm,
            "source": "llm",
        }
        return sections, form_dict, meta
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM form failed for %s (%s); DSP fallback", title, exc)
        return resolve_sections_dsp(
            file_path,
            title,
            bpm,
            vocals_path=vocals_path,
            measured_duration_sec=measured,
        )


def resolve_sections_dsp(
    file_path: str,
    title: str,
    bpm: float,
    vocals_path: str | None = None,
    *,
    measured_duration_sec: float | None = None,
    meter_numerator: int = 4,
) -> tuple[list[Section], dict[str, Any] | None, dict[str, Any]]:
    """Heuristic librosa ``detect_sections`` only."""
    del measured_duration_sec
    sections = detect_sections(
        file_path,
        vocals_path=vocals_path,
        max_sections=None,
        meter_numerator=meter_numerator,
        bpm=bpm,
    )
    form_dict: dict[str, Any] = {
        "title": title,
        "bpm": bpm,
        "time_signature_numerator": meter_numerator,
        "time_signature_denominator": 4,
        "style_notes": "structure via dsp",
        "total_duration_sec": sections[-1].end_sec if sections else 0.0,
        "sections": [
            {
                "name": s.name or s.label,
                "label": s.label,
                "start_sec": s.start_sec,
                "end_sec": s.end_sec,
                "approx_beats": s.approx_beats,
                "description": s.description,
            }
            for s in sections
        ],
    }
    meta = {
        "structure_source": "dsp",
        "structure_mode": "dsp",
        "bpm": bpm,
        "meter_numerator": meter_numerator,
        "meter_denominator": 4,
        "title": title,
        "measured_bpm": bpm,
        "source": "dsp",
        "sections": [s.to_prompt_dict() for s in sections],
    }
    return sections, form_dict, meta
