"""Song structure resolution — allin1 primary; DSP fallback.

LLM form analysis was removed: title-based timestamps were unreliable.
Prefer :mod:`services.allin1_structure` for new code.
"""

from __future__ import annotations

from typing import Any

from services.allin1_structure import (
    StructureBundle,
    analyze_structure,
    resolve_sections_for_song,
)
from services.structure import Section

__all__ = [
    "Section",
    "StructureBundle",
    "analyze_structure",
    "resolve_sections_for_song",
]

# Soft stub so accidental SongFormAnalysis imports don't crash.
SongFormAnalysis = dict[str, Any]
