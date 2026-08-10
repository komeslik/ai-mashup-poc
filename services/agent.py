"""LLM-driven mashup strategy via OpenAI or Gemini structured outputs."""

from __future__ import annotations

import logging
import os
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

SongSource = Literal["song_a", "song_b"]
ProviderName = Literal["openai", "gemini"]

SYSTEM_PROMPT = (
    "You are a DJ / music producer assistant. Given two songs' BPMs, decide a "
    "simple mashup strategy. Vocals and instrumental MUST come from different "
    "songs. Prefer matching vocals to the instrumental's BPM as target_bpm. "
    "stretch_factor MUST equal target_bpm / vocal_source_bpm."
)


class MashupDecision(BaseModel):
    """Structured mixing strategy for a two-song mashup."""

    vocal_source: SongSource = Field(
        description="Which song supplies the vocals stem."
    )
    instrumental_source: SongSource = Field(
        description="Which song supplies the instrumental stem."
    )
    # Avoid Field(gt=0): Gemini's Schema rejects JSON Schema exclusiveMinimum.
    target_bpm: float = Field(description="BPM to align the mashup to.")
    stretch_factor: float = Field(
        description="Time-stretch factor applied to vocals: target_bpm / vocal_source_bpm.",
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


def _fallback_decision(song_a_bpm: float, song_b_bpm: float) -> MashupDecision:
    """Deterministic POC fallback: A vocals over B instrumental."""
    target_bpm = float(song_b_bpm)
    stretch_factor = target_bpm / float(song_a_bpm)
    return MashupDecision(
        vocal_source="song_a",
        instrumental_source="song_b",
        target_bpm=target_bpm,
        stretch_factor=stretch_factor,
    )


def _normalize_stretch(
    decision: MashupDecision,
    song_a_bpm: float,
    song_b_bpm: float,
) -> MashupDecision:
    """Recompute stretch_factor from BPMs so the DSP step stays consistent."""
    vocal_bpm = song_a_bpm if decision.vocal_source == "song_a" else song_b_bpm
    stretch_factor = decision.target_bpm / vocal_bpm
    return decision.model_copy(update={"stretch_factor": stretch_factor})


def _user_prompt(song_a_bpm: float, song_b_bpm: float) -> str:
    return (
        f"Song A BPM: {song_a_bpm:.3f}\n"
        f"Song B BPM: {song_b_bpm:.3f}\n"
        "Return a structured mashup decision."
    )


def _resolve_provider() -> ProviderName:
    raw = (os.getenv("LLM_PROVIDER") or "gemini").strip().lower()
    if raw not in ("openai", "gemini"):
        logger.warning("Unknown LLM_PROVIDER=%r; defaulting to gemini", raw)
        return "gemini"
    return raw  # type: ignore[return-value]


def _decide_with_openai(song_a_bpm: float, song_b_bpm: float) -> MashupDecision:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")

    client = OpenAI(api_key=api_key)
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(song_a_bpm, song_b_bpm)},
        ],
        response_format=MashupDecision,
        temperature=0.2,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("OpenAI returned an empty structured response")
    return _normalize_stretch(parsed, song_a_bpm, song_b_bpm)


def _decide_with_gemini(song_a_bpm: float, song_b_bpm: float) -> MashupDecision:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=_user_prompt(song_a_bpm, song_b_bpm),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
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


def decide_mashup_strategy(song_a_bpm: float, song_b_bpm: float) -> MashupDecision:
    """
    Ask an LLM for a mashup strategy, falling back to a fixed rule on failure.

    Provider is selected with ``LLM_PROVIDER`` (``openai`` or ``gemini``).
    Prefer complementary sources (vocals from one song, instrumental from the other)
    and a sensible target BPM (typically the instrumental's BPM).
    """
    if song_a_bpm <= 0 or song_b_bpm <= 0:
        raise ValueError("BPM values must be positive")

    provider = _resolve_provider()
    logger.info("Mashup strategy provider: %s", provider)

    try:
        if provider == "openai":
            return _decide_with_openai(song_a_bpm, song_b_bpm)
        return _decide_with_gemini(song_a_bpm, song_b_bpm)
    except Exception as exc:  # noqa: BLE001 — keep POC resilient overnight
        logger.warning(
            "%s mashup decision failed (%s); using fallback",
            provider,
            exc,
        )
        return _fallback_decision(song_a_bpm, song_b_bpm)
