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
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from services.agent import MashupDecision, decide_arrangement, decide_mashup_strategy
from services.audio import get_bpm, get_key, minimal_meeting_shifts
from services.demucs import DemucsError, ensure_allin1_demix_layout, separate_full_stems
from services.dual_mix import (
    CreativeMode,
    PhraseVocalPolicy,
    build_bassline_mashup,
    build_dual_vocal_mashup,
)
from services.form_analysis import resolve_sections_dsp, resolve_sections_llm
from services.mashability import MashabilityWeights
from services.studio_mix import (
    apply_committed_sections,
    ensure_song_preview,
    extract_song_range,
    load_studio,
    render_studio,
    save_studio,
)

load_dotenv()

# Hugging Face auth for allin1 / hub downloads.
_hf = (os.getenv("HF_TOKEN") or os.getenv("HF_TOKEM") or "").strip()
if _hf:
    os.environ["HF_TOKEN"] = _hf
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _hf)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if _hf:
    logger.info("HF token present (Hugging Face downloads authenticated)")
else:
    logger.info("HF token absent (public HF downloads only)")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SESSIONS_ROOT = Path("/tmp/mashup_sessions")
SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AI Song Mashup API", version="0.3.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

VocalPolicyForm = Literal[
    "auto",
    "alternate",
    "a_lead_b_harmony",
    "b_lead_a_harmony",
]
StructureMode = Literal["allin1", "llm", "dsp"]


def _allow_harmony(vocal_policy: VocalPolicyForm, _policy: PhraseVocalPolicy | None = None) -> bool:
    """Director-strict by default; muted harmony only when UI picks a harmony policy."""
    return vocal_policy in ("a_lead_b_harmony", "b_lead_a_harmony")


class StudioPutBody(BaseModel):
    studio: dict[str, Any] = Field(default_factory=dict)


class StudioPreviewBody(BaseModel):
    column_id: str | None = None


class CommitSectionsBody(BaseModel):
    sections_a: list[dict[str, Any]] | None = None
    sections_b: list[dict[str, Any]] | None = None


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


def _session_dir(session_id: str) -> Path:
    session = SESSIONS_ROOT / session_id
    if not (session / "metadata.json").is_file():
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _resolve_structure(
    mode: StructureMode,
    *,
    file_path: str,
    title: str,
    bpm: float,
    vocals_path: str | None,
    work_dir: Path,
    demix_dir: Path | None,
) -> tuple[list, dict | None, dict]:
    if mode == "llm":
        return resolve_sections_llm(
            file_path,
            title,
            bpm,
            vocals_path,
            measured_duration_sec=None,
        )
    if mode == "dsp":
        return resolve_sections_dsp(
            file_path,
            title,
            bpm,
            vocals_path,
        )
    # allin1 (lazy import so missing natten still allows llm/dsp modes)
    from services.allin1_structure import resolve_sections_for_song

    return resolve_sections_for_song(
        file_path,
        title,
        bpm,
        vocals_path,
        work_dir=work_dir,
        demix_dir=demix_dir,
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/mashup/sessions/{session_id}")
def get_session(session_id: str) -> JSONResponse:
    session = _session_dir(session_id)
    meta = json.loads((session / "metadata.json").read_text(encoding="utf-8"))
    return JSONResponse(meta)


@app.get("/api/mashup/sessions/{session_id}/mashup")
def get_session_mashup(session_id: str) -> FileResponse:
    session = _session_dir(session_id)
    path = session / "mashup.mp3"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Session mashup missing")
    return FileResponse(path=str(path), media_type="audio/mpeg", filename="mashup.mp3")


@app.get("/api/mashup/sessions/{session_id}/studio")
def get_studio(session_id: str) -> JSONResponse:
    session = _session_dir(session_id)
    try:
        studio = load_studio(session)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(studio)


@app.put("/api/mashup/sessions/{session_id}/studio")
def put_studio(session_id: str, body: StudioPutBody) -> JSONResponse:
    session = _session_dir(session_id)
    if not isinstance(body.studio, dict) or "columns" not in body.studio:
        raise HTTPException(status_code=400, detail="studio.columns required")
    save_studio(session, body.studio)
    return JSONResponse({"ok": True})


@app.post("/api/mashup/sessions/{session_id}/studio/commit-sections")
def commit_sections(session_id: str, body: CommitSectionsBody) -> JSONResponse:
    session = _session_dir(session_id)
    try:
        studio = load_studio(session)
        studio = apply_committed_sections(
            studio,
            sections_a=body.sections_a,
            sections_b=body.sections_b,
        )
        save_studio(session, studio)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(studio)


@app.get("/api/mashup/sessions/{session_id}/song/{song}/audio")
async def get_song_audio(
    session_id: str,
    song: Literal["a", "b"],
) -> FileResponse:
    session = _session_dir(session_id)
    try:
        path = await asyncio.to_thread(ensure_song_preview, session, song)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(
        path=str(path),
        media_type="audio/mpeg",
        filename=f"song_{song}_preview.mp3",
    )


@app.get("/api/mashup/sessions/{session_id}/song/{song}/section-preview")
async def get_song_section_preview(
    session_id: str,
    song: Literal["a", "b"],
    start: float = 0.0,
    end: float = 1.0,
) -> FileResponse:
    session = _session_dir(session_id)
    out_dir = tempfile.mkdtemp(prefix="sec_prev_", dir="/tmp")
    out_path = Path(out_dir) / "section.mp3"
    try:
        await asyncio.to_thread(
            extract_song_range, session, song, start, end, out_path
        )
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(out_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(
        path=str(out_path),
        media_type="audio/mpeg",
        filename=f"song_{song}_section.mp3",
        background=BackgroundTask(shutil.rmtree, out_dir, ignore_errors=True),
    )


@app.get("/api/mashup/sessions/{session_id}/clips/{song}/{stem}")
async def get_clip(
    session_id: str,
    song: Literal["a", "b"],
    stem: Literal["vocals", "drums", "bass", "other"],
    section: int = 0,
) -> FileResponse:
    from services.audio import extract_audio_segment
    from pydub import AudioSegment

    session = _session_dir(session_id)
    studio = load_studio(session)
    sections = studio.get("sections_a" if song == "a" else "sections_b") or []
    if not sections:
        raise HTTPException(status_code=400, detail="No sections for song")
    idx = int(section) % len(sections)
    sec = sections[idx]
    start = float(sec.get("start_sec", 0.0))
    end = float(sec.get("end_sec", start + 1.0))
    stem_dir = session / f"stems_{song}"
    src = None
    for ext in ("wav", "mp3"):
        candidate = stem_dir / f"{stem}.{ext}"
        if candidate.is_file():
            src = candidate
            break
    if src is None:
        raise HTTPException(status_code=404, detail=f"Stem missing: {song}/{stem}")

    out_dir = tempfile.mkdtemp(prefix="clip_", dir="/tmp")
    wav_out = Path(out_dir) / "clip.wav"
    mp3_out = Path(out_dir) / "clip.mp3"
    try:
        await asyncio.to_thread(
            extract_audio_segment, str(src), str(wav_out), start, end
        )
        await asyncio.to_thread(
            lambda: AudioSegment.from_file(wav_out).export(str(mp3_out), format="mp3")
        )
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(out_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(
        path=str(mp3_out),
        media_type="audio/mpeg",
        filename=f"{song}_{stem}_s{idx}.mp3",
        background=BackgroundTask(shutil.rmtree, out_dir, ignore_errors=True),
    )


@app.post("/api/mashup/sessions/{session_id}/studio/preview")
async def studio_preview(session_id: str, body: StudioPreviewBody) -> FileResponse:
    session = _session_dir(session_id)
    out_dir = tempfile.mkdtemp(prefix="studio_prev_", dir="/tmp")
    out_path = Path(out_dir) / "preview.mp3"
    try:
        studio = load_studio(session)
        await asyncio.to_thread(
            render_studio,
            session,
            studio,
            column_id=body.column_id,
            output_path=out_path,
        )
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(out_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(
        path=str(out_path),
        media_type="audio/mpeg",
        filename="preview.mp3",
        background=BackgroundTask(shutil.rmtree, out_dir, ignore_errors=True),
    )


@app.post("/api/mashup/sessions/{session_id}/studio/render")
async def studio_render(session_id: str) -> FileResponse:
    session = _session_dir(session_id)
    out_path = session / "mashup-edit.mp3"
    try:
        studio = load_studio(session)
        await asyncio.to_thread(
            render_studio, session, studio, output_path=out_path
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(
        path=str(out_path),
        media_type="audio/mpeg",
        filename="mashup-edit.mp3",
    )


@app.post("/api/mashup")
async def create_mashup(
    song_a: UploadFile = File(..., description="First song audio file"),
    song_b: UploadFile = File(..., description="Second song audio file"),
    vocal_policy: VocalPolicyForm = Form("alternate"),
    creative_mode: CreativeMode = Form("forced_match"),
    structure_mode: StructureMode = Form("allin1"),
    harmonic_weight: float = Form(0.6),
    rhythmic_weight: float = Form(0.25),
    spectral_weight: float = Form(0.15),
) -> FileResponse:
    """Separate stems, dual-vocal (or bassline) mashup, return MP3 + metadata headers."""
    if min(harmonic_weight, rhythmic_weight, spectral_weight) < 0:
        raise HTTPException(status_code=400, detail="Mashability weights must be >= 0")
    if harmonic_weight + rhythmic_weight + spectral_weight <= 0:
        raise HTTPException(status_code=400, detail="At least one mashability weight must be > 0")
    if structure_mode not in ("allin1", "llm", "dsp"):
        raise HTTPException(status_code=400, detail="Invalid structure_mode")

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

            demix_root = work / "demix"
            demix_root.mkdir()
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
                    form_a = form_b = None
                    form_meta_a = form_meta_b = {}
                    meter_a = meter_b = 4
                    shift_a = shift_b = 0
                    key_a = key_b = meeting_pc = None
                    full_a = full_b = None
                    logger.info("Mashup decision: %s", decision.model_dump())
                    first_lead = "a" if decision.vocal_source == "song_a" else "b"
                    policy = _resolve_policy(vocal_policy, decision)
                else:
                    logger.info(
                        "Full Demucs WAV stems (shared demix) structure_mode=%s",
                        structure_mode,
                    )
                    full_a = await asyncio.to_thread(
                        separate_full_stems, str(song_a_path), str(demix_root)
                    )
                    full_b = await asyncio.to_thread(
                        separate_full_stems, str(song_b_path), str(demix_root)
                    )
                    vocals_a = full_a.vocals
                    vocals_b = full_b.vocals
                    instr_a = full_a.instrumental

                    # Ensure allin1 demix cache layout matches analyze input basenames.
                    ensure_allin1_demix_layout(demix_root, song_a_path.stem, full_a)
                    ensure_allin1_demix_layout(demix_root, song_b_path.stem, full_b)

                    struct_dir = work / "structure"
                    struct_dir.mkdir(exist_ok=True)
                    shared_demix = demix_root if structure_mode == "allin1" else None

                    sections_a, form_a, form_meta_a = await asyncio.to_thread(
                        _resolve_structure,
                        structure_mode,
                        file_path=str(song_a_path),
                        title=title_a,
                        bpm=bpm_a,
                        vocals_path=vocals_a,
                        work_dir=struct_dir / "a",
                        demix_dir=shared_demix,
                    )
                    sections_b, form_b, form_meta_b = await asyncio.to_thread(
                        _resolve_structure,
                        structure_mode,
                        file_path=str(song_b_path),
                        title=title_b,
                        bpm=bpm_b,
                        vocals_path=vocals_b,
                        work_dir=struct_dir / "b",
                        demix_dir=shared_demix,
                    )
                    form_meta_a = dict(form_meta_a or {})
                    form_meta_b = dict(form_meta_b or {})
                    form_meta_a["structure_mode"] = structure_mode
                    form_meta_b["structure_mode"] = structure_mode

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
                        "Sections — A: %d (%s bpm=%.2f), B: %d (%s bpm=%.2f) mode=%s",
                        len(sections_a),
                        form_meta_a.get("structure_source") or form_meta_a.get("source"),
                        bpm_a,
                        len(sections_b),
                        form_meta_b.get("structure_source") or form_meta_b.get("source"),
                        bpm_b,
                        structure_mode,
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
                        separate_full_stems, str(song_a_path), str(demix_root)
                    )
                    full_b = await asyncio.to_thread(
                        separate_full_stems, str(song_b_path), str(demix_root)
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
                        drums_a=full_a.drums,
                        bass_a=full_a.bass,
                        other_a=full_a.other,
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
        metadata["structure_mode"] = structure_mode
        (session_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
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
