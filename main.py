"""
AI Song Mashup API — overnight POC.

Run:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from services.agent import MashupDecision, decide_arrangement, decide_mashup_strategy
from services.audio import get_bpm, get_key, minimal_meeting_shifts
from services.allin1_structure import resolve_sections_for_song
from services.demucs import DemucsError, separate_full_stems
from services.dual_mix import (
    CreativeMode,
    PhraseVocalPolicy,
    build_bassline_mashup,
    build_dual_vocal_mashup,
    reassemble_session,
)
from services.library import list_library_tracks, rank_library_against_query
from services.mashability import MashabilityWeights

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SESSIONS_ROOT = Path("/tmp/mashup_sessions")
SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AI Song Mashup API", version="0.2.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

VocalPolicyForm = Literal[
    "auto",
    "alternate",
    "a_lead_b_harmony",
    "b_lead_a_harmony",
]


def _allow_harmony(vocal_policy: VocalPolicyForm, _policy: PhraseVocalPolicy | None = None) -> bool:
    """Director-strict by default; muted harmony only when UI picks a harmony policy."""
    return vocal_policy in ("a_lead_b_harmony", "b_lead_a_harmony")


class ReassembleBody(BaseModel):
    enabled_indices: list[int] = Field(default_factory=list)


def _suffix_for_upload(upload: UploadFile, default: str = ".mp3") -> str:
    name = upload.filename or ""
    suffix = Path(name).suffix
    return suffix if suffix else default


def _save_upload(upload: UploadFile, destination: Path) -> Path:
    with destination.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return destination


def _resolve_policy(
    vocal_policy: VocalPolicyForm,
    decision: MashupDecision,
) -> PhraseVocalPolicy:
    if vocal_policy == "auto":
        return decision.phrase_vocal_policy
    return vocal_policy


def _metadata_header(metadata: dict) -> str:
    raw = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/library")
def api_library() -> dict:
    tracks = list_library_tracks(BASE_DIR)
    return {
        "library_dir": str((BASE_DIR / "library").resolve()),
        "tracks": [
            {"id": t.id, "name": t.name, "bpm": t.bpm} for t in tracks
        ],
    }


@app.post("/api/library/search")
async def api_library_search(
    query: UploadFile = File(..., description="Query track to rank library against"),
    top_k: int = Form(5),
) -> JSONResponse:
    with tempfile.TemporaryDirectory(prefix="lib_query_", dir="/tmp") as tmp:
        path = Path(tmp) / f"query{_suffix_for_upload(query)}"
        _save_upload(query, path)
        ranked = await asyncio.to_thread(
            rank_library_against_query,
            str(path),
            BASE_DIR,
            top_k=top_k,
        )
    return JSONResponse({"results": ranked})


@app.get("/api/mashup/sessions/{session_id}")
def get_session(session_id: str) -> JSONResponse:
    session = SESSIONS_ROOT / session_id
    meta = session / "metadata.json"
    if not meta.is_file():
        raise HTTPException(status_code=404, detail="Session not found")
    return JSONResponse(json.loads(meta.read_text(encoding="utf-8")))


@app.post("/api/mashup/sessions/{session_id}/reassemble")
async def reassemble(session_id: str, body: ReassembleBody) -> FileResponse:
    session = SESSIONS_ROOT / session_id
    if not (session / "metadata.json").is_file():
        raise HTTPException(status_code=404, detail="Session not found")

    out_dir = tempfile.mkdtemp(prefix="mashup_edit_", dir="/tmp")
    out_path = Path(out_dir) / "mashup.mp3"
    try:
        await asyncio.to_thread(
            reassemble_session,
            session,
            body.enabled_indices,
            str(out_path),
        )
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(out_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return FileResponse(
        path=str(out_path),
        media_type="audio/mpeg",
        filename="mashup.mp3",
        background=BackgroundTask(shutil.rmtree, out_dir, ignore_errors=True),
    )


@app.post("/api/mashup")
async def create_mashup(
    song_a: UploadFile = File(..., description="First song audio file"),
    song_b: UploadFile = File(..., description="Second song audio file"),
    vocal_policy: VocalPolicyForm = Form("alternate"),
    creative_mode: CreativeMode = Form("forced_match"),
    harmonic_weight: float = Form(0.6),
    rhythmic_weight: float = Form(0.25),
    spectral_weight: float = Form(0.15),
) -> FileResponse:
    """Separate stems, dual-vocal (or bassline) mashup, return MP3 + metadata headers."""
    if min(harmonic_weight, rhythmic_weight, spectral_weight) < 0:
        raise HTTPException(status_code=400, detail="Mashability weights must be >= 0")
    if harmonic_weight + rhythmic_weight + spectral_weight <= 0:
        raise HTTPException(status_code=400, detail="At least one mashability weight must be > 0")

    weights = MashabilityWeights(
        harmonic=harmonic_weight,
        rhythmic=rhythmic_weight,
        spectral=spectral_weight,
    )

    final_dir = tempfile.mkdtemp(prefix="mashup_out_", dir="/tmp")
    final_path = Path(final_dir) / "mashup.mp3"
    session_id = uuid.uuid4().hex
    session_dir = SESSIONS_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(prefix="mashup_work_", dir="/tmp") as work_dir:
            work = Path(work_dir)
            song_a_path = work / f"song_a{_suffix_for_upload(song_a)}"
            song_b_path = work / f"song_b{_suffix_for_upload(song_b)}"

            try:
                _save_upload(song_a, song_a_path)
                _save_upload(song_b, song_b_path)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=400, detail=f"Failed to save uploads: {exc}") from exc

            stems_a_dir = work / "stems_a"
            stems_b_dir = work / "stems_b"
            stems_a_dir.mkdir()
            stems_b_dir.mkdir()

            title_a = Path(song_a.filename or "Song A").stem
            title_b = Path(song_b.filename or "Song B").stem

            try:
                bpm_a = get_bpm(str(song_a_path))
                bpm_b = get_bpm(str(song_b_path))
                logger.info("Detected BPM — A: %.2f, B: %.2f", bpm_a, bpm_b)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=400,
                    detail=f"BPM detection failed: {exc}",
                ) from exc

            try:
                if creative_mode == "bassline":
                    decision = decide_mashup_strategy(
                        bpm_a,
                        bpm_b,
                        creative_mode=creative_mode,
                    )
                    blueprint = None
                    sections_a = None
                    sections_b = None
                    logger.info("Mashup decision: %s", decision.model_dump())
                    first_lead = "a" if decision.vocal_source == "song_a" else "b"
                    policy = _resolve_policy(vocal_policy, decision)
                else:
                    logger.info("Full Demucs stems for Song A anchor mashup")
                    full_a = await asyncio.to_thread(
                        separate_full_stems, str(song_a_path), str(stems_a_dir)
                    )
                    full_b = await asyncio.to_thread(
                        separate_full_stems, str(song_b_path), str(stems_b_dir)
                    )
                    vocals_a = full_a.vocals
                    vocals_b = full_b.vocals
                    instr_a = full_a.instrumental

                    # allin1 (or DSP fallback) structure on original uploads.
                    struct_dir = work / "structure"
                    struct_dir.mkdir(exist_ok=True)
                    sections_a, form_a, form_meta_a = await asyncio.to_thread(
                        resolve_sections_for_song,
                        str(song_a_path),
                        title_a,
                        bpm_a,
                        vocals_a,
                        work_dir=struct_dir / "a",
                    )
                    sections_b, form_b, form_meta_b = await asyncio.to_thread(
                        resolve_sections_for_song,
                        str(song_b_path),
                        title_b,
                        bpm_b,
                        vocals_b,
                        work_dir=struct_dir / "b",
                    )
                    # Prefer allin1 BPM when available.
                    if form_meta_a.get("structure_source") == "allin1" and form_meta_a.get(
                        "bpm"
                    ):
                        bpm_a = float(form_meta_a["bpm"])
                    if form_meta_b.get("structure_source") == "allin1" and form_meta_b.get(
                        "bpm"
                    ):
                        bpm_b = float(form_meta_b["bpm"])
                    meter_a = int(form_meta_a.get("meter_numerator") or 4)
                    meter_b = int(form_meta_b.get("meter_numerator") or 4)
                    logger.info(
                        "Sections — A: %d (%s bpm=%.2f), B: %d (%s bpm=%.2f)",
                        len(sections_a),
                        form_meta_a.get("structure_source") or form_meta_a.get("source"),
                        bpm_a,
                        len(sections_b),
                        form_meta_b.get("structure_source") or form_meta_b.get("source"),
                        bpm_b,
                    )

                    try:
                        key_a = await asyncio.to_thread(get_key, str(song_a_path))
                        key_b = await asyncio.to_thread(get_key, str(song_b_path))
                        shift_a, shift_b, meeting_pc = minimal_meeting_shifts(
                            key_a, key_b
                        )
                        logger.info(
                            "Key meeting — A %s / B %s → pc=%d shifts A%+d B%+d",
                            key_a,
                            key_b,
                            meeting_pc,
                            shift_a,
                            shift_b,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Key meeting failed (%s); shifts=0", exc)
                        key_a, key_b, meeting_pc = None, None, None
                        shift_a, shift_b = 0, 0

                    allow_harmony = _allow_harmony(vocal_policy)

                    blueprint = decide_arrangement(
                        bpm_a,
                        bpm_b,
                        sections_a,
                        sections_b,
                        creative_mode=creative_mode,
                        title_a=title_a,
                        title_b=title_b,
                        allow_harmony=allow_harmony,
                    )
                    decision = blueprint.as_decision()
                    first_lead = "a" if decision.vocal_source == "song_a" else "b"
                    policy = _resolve_policy(vocal_policy, decision)
                    if not allow_harmony:
                        policy = "alternate"
                    logger.info(
                        "Arrangement: %s | timeline=%d | actions=%d | strict=%s",
                        blueprint.arranging_reasoning,
                        len(blueprint.timeline),
                        len(blueprint.actions),
                        not allow_harmony,
                    )
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=502,
                    detail=f"Mashup strategy failed: {exc}",
                ) from exc

            try:
                if creative_mode == "bassline":
                    logger.info("Full Demucs stems for bassline mode")
                    full_a = await asyncio.to_thread(
                        separate_full_stems, str(song_a_path), str(stems_a_dir)
                    )
                    full_b = await asyncio.to_thread(
                        separate_full_stems, str(song_b_path), str(stems_b_dir)
                    )
                    if decision.instrumental_source == "song_a":
                        drums_bed, other_bed = full_a.drums, full_a.other
                    else:
                        drums_bed, other_bed = full_b.drums, full_b.other
                    result = await asyncio.to_thread(
                        build_bassline_mashup,
                        bass_a=full_a.bass,
                        bass_b=full_b.bass,
                        drums_bed=drums_bed,
                        other_bed=other_bed,
                        bpm_a=bpm_a,
                        bpm_b=bpm_b,
                        target_bpm=decision.target_bpm,
                        work_dir=work / "bass",
                        output_path=str(final_path),
                        first_lead=first_lead,
                        weights=weights,
                        session_dir=session_dir,
                    )
                else:
                    # Song A is always the instrumental anchor bed.
                    result = await asyncio.to_thread(
                        build_dual_vocal_mashup,
                        vocals_a=vocals_a,
                        vocals_b=vocals_b,
                        instrumental=instr_a,
                        bpm_a=bpm_a,
                        bpm_b=bpm_b,
                        target_bpm=decision.target_bpm,
                        work_dir=work / "dual",
                        output_path=str(final_path),
                        policy=policy,
                        first_lead=first_lead,
                        weights=weights,
                        creative_mode=creative_mode,
                        session_dir=session_dir,
                        blueprint=blueprint,
                        sections_a=sections_a,
                        sections_b=sections_b,
                        shift_a=shift_a,
                        shift_b=shift_b,
                        key_a=key_a,
                        key_b=key_b,
                        meeting_pc=meeting_pc,
                        drums_b=full_b.drums,
                        bass_b=full_b.bass,
                        other_b=full_b.other,
                        form_a=form_a if isinstance(form_a, dict) else form_meta_a,
                        form_b=form_b if isinstance(form_b, dict) else form_meta_b,
                        structure_meta_a=form_meta_a,
                        structure_meta_b=form_meta_b,
                        meter_a=meter_a,
                        meter_b=meter_b,
                    )
            except DemucsError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            except Exception as exc:  # noqa: BLE001
                logger.exception("Mashup build failed")
                raise HTTPException(
                    status_code=500,
                    detail=f"Mashup failed: {exc}",
                ) from exc

        if not final_path.is_file():
            raise HTTPException(status_code=500, detail="Mashup file was not produced")

        metadata = dict(result.metadata)
        metadata["session_id"] = session_id
        headers = {
            "X-Mashup-Session-Id": session_id,
            "X-Mashup-Metadata": _metadata_header(metadata),
            "Access-Control-Expose-Headers": "X-Mashup-Session-Id, X-Mashup-Metadata",
        }

        return FileResponse(
            path=str(final_path),
            media_type="audio/mpeg",
            filename="mashup.mp3",
            headers=headers,
            background=BackgroundTask(shutil.rmtree, final_dir, ignore_errors=True),
        )
    except HTTPException:
        shutil.rmtree(final_dir, ignore_errors=True)
        shutil.rmtree(session_dir, ignore_errors=True)
        raise
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(final_dir, ignore_errors=True)
        shutil.rmtree(session_dir, ignore_errors=True)
        logger.exception("Unhandled mashup error")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc
