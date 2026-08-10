"""
AI Song Mashup API — overnight POC.

Run:
    uvicorn main:app --reload --port 8000

POST multipart form fields ``song_a`` and ``song_b`` to ``/api/mashup``.
Requires ffmpeg on PATH, local ``demucs`` CLI, and OPENAI_API_KEY in ``.env``.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from services.agent import MashupDecision, decide_mashup_strategy
from services.audio import get_bpm, mix_stems, time_stretch_audio
from services.demucs import DemucsError, separate_stems

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="AI Song Mashup API", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

STRETCH_EPSILON = 0.01


def _suffix_for_upload(upload: UploadFile, default: str = ".mp3") -> str:
    name = upload.filename or ""
    suffix = Path(name).suffix
    return suffix if suffix else default


def _save_upload(upload: UploadFile, destination: Path) -> Path:
    with destination.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return destination


def _select_stems(
    decision: MashupDecision,
    stems_a: tuple[str, str],
    stems_b: tuple[str, str],
) -> tuple[str, str]:
    vocals_a, instrumental_a = stems_a
    vocals_b, instrumental_b = stems_b

    vocal_path = vocals_a if decision.vocal_source == "song_a" else vocals_b
    instrumental_path = (
        instrumental_a if decision.instrumental_source == "song_a" else instrumental_b
    )
    return vocal_path, instrumental_path


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/mashup")
async def create_mashup(
    song_a: UploadFile = File(..., description="First song audio file"),
    song_b: UploadFile = File(..., description="Second song audio file"),
) -> FileResponse:
    """
    Separate stems, decide a mix strategy, time-stretch vocals, and return an MP3.
    """
    # Persist the final mashup outside the working TemporaryDirectory so
    # FileResponse can stream it after workspace cleanup.
    final_dir = tempfile.mkdtemp(prefix="mashup_out_", dir="/tmp")
    final_path = Path(final_dir) / "mashup.mp3"

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

            try:
                logger.info("Separating stems for song A (local Demucs)")
                stems_a = await asyncio.to_thread(
                    separate_stems, str(song_a_path), str(stems_a_dir)
                )
                logger.info("Separating stems for song B (local Demucs)")
                stems_b = await asyncio.to_thread(
                    separate_stems, str(song_b_path), str(stems_b_dir)
                )
            except DemucsError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

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
                decision = decide_mashup_strategy(bpm_a, bpm_b)
                logger.info("Mashup decision: %s", decision.model_dump())
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=502,
                    detail=f"Mashup strategy failed: {exc}",
                ) from exc

            vocal_path, instrumental_path = _select_stems(decision, stems_a, stems_b)

            stretched_vocals = work / "vocals_stretched.wav"
            if abs(decision.stretch_factor - 1.0) < STRETCH_EPSILON:
                vocals_for_mix = vocal_path
            else:
                try:
                    vocals_for_mix = time_stretch_audio(
                        vocal_path,
                        str(stretched_vocals),
                        decision.stretch_factor,
                    )
                except Exception as exc:  # noqa: BLE001
                    raise HTTPException(
                        status_code=500,
                        detail=f"Time-stretch failed: {exc}",
                    ) from exc

            try:
                mix_stems(vocals_for_mix, instrumental_path, str(final_path))
            except RuntimeError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=500,
                    detail=f"Mixing failed: {exc}",
                ) from exc

        if not final_path.is_file():
            raise HTTPException(status_code=500, detail="Mashup file was not produced")

        return FileResponse(
            path=str(final_path),
            media_type="audio/mpeg",
            filename="mashup.mp3",
            background=BackgroundTask(shutil.rmtree, final_dir, ignore_errors=True),
        )
    except HTTPException:
        shutil.rmtree(final_dir, ignore_errors=True)
        raise
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(final_dir, ignore_errors=True)
        logger.exception("Unhandled mashup error")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc
