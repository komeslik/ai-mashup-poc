"""LLM-driven mashup strategy via OpenAI structured outputs."""

from __future__ import annotations

import logging
import os
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

SongSource = Literal["song_a", "song_b"]


class MashupDecision(BaseModel):
    """Structured mixing strategy for a two-song mashup."""

    vocal_source: SongSource = Field(
        description="Which song supplies the vocals stem."
    )
    instrumental_source: SongSource = Field(
        description="Which song supplies the instrumental stem."
    )
    target_bpm: float = Field(gt=0, description="BPM to align the mashup to.")
    stretch_factor: float = Field(
        gt=0,
        description="Time-stretch factor applied to vocals: target_bpm / vocal_source_bpm.",
    )

    @model_validator(mode="after")
    def sources_must_differ(self) -> MashupDecision:
        if self.vocal_source == self.instrumental_source:
            raise ValueError(
                "vocal_source and instrumental_source must come from different songs"
            )
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


def decide_mashup_strategy(song_a_bpm: float, song_b_bpm: float) -> MashupDecision:
    """
    Ask an LLM for a mashup strategy, falling back to a fixed rule on failure.

    Prefer complementary sources (vocals from one song, instrumental from the other)
    and a sensible target BPM (typically the instrumental's BPM).
    """
    if song_a_bpm <= 0 or song_b_bpm <= 0:
        raise ValueError("BPM values must be positive")

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        logger.warning("OPENAI_API_KEY missing; using deterministic mashup fallback")
        return _fallback_decision(song_a_bpm, song_b_bpm)

    client = OpenAI(api_key=api_key)

    system_prompt = (
        "You are a DJ / music producer assistant. Given two songs' BPMs, decide a "
        "simple mashup strategy. Vocals and instrumental MUST come from different "
        "songs. Prefer matching vocals to the instrumental's BPM as target_bpm. "
        "stretch_factor MUST equal target_bpm / vocal_source_bpm."
    )
    user_prompt = (
        f"Song A BPM: {song_a_bpm:.3f}\n"
        f"Song B BPM: {song_b_bpm:.3f}\n"
        "Return a structured mashup decision."
    )

    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=MashupDecision,
            temperature=0.2,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("OpenAI returned an empty structured response")
        return _normalize_stretch(parsed, song_a_bpm, song_b_bpm)
    except Exception as exc:  # noqa: BLE001 — keep POC resilient overnight
        logger.warning("OpenAI mashup decision failed (%s); using fallback", exc)
        return _fallback_decision(song_a_bpm, song_b_bpm)
