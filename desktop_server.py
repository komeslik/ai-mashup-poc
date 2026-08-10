"""Packaged / Electron entrypoint for the mashup FastAPI server.

Dev (localhost unchanged):
    uvicorn main:app --reload --port 8000

Desktop / sidecar:
    python desktop_server.py --port 8765
    # or the PyInstaller binary: mashup-server.exe --port 8765
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _app_data_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "ai-mashup-poc"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "ai-mashup-poc"
    return Path.home() / ".config" / "ai-mashup-poc"


def _ensure_user_env(app_dir: Path) -> Path:
    """Create a user .env from the example template if missing."""
    app_dir.mkdir(parents=True, exist_ok=True)
    env_path = app_dir / ".env"
    if env_path.is_file():
        return env_path

    # Prefer bundled example next to frozen exe / project root.
    candidates = [
        Path(getattr(sys, "_MEIPASS", "")) / ".env.example",
        Path(__file__).resolve().parent / ".env.example",
    ]
    template = (
        "LLM_PROVIDER=gemini\n"
        "GEMINI_API_KEY=\n"
        "GEMINI_MODEL=gemini-flash-latest\n"
        "OPENAI_API_KEY=\n"
        "OPENAI_MODEL=gpt-4o-mini\n"
        "HF_TOKEN=\n"
    )
    for c in candidates:
        if c.is_file():
            template = c.read_text(encoding="utf-8")
            break
    env_path.write_text(template, encoding="utf-8")
    return env_path


def _prepare_env(port: int) -> None:
    app_dir = _app_data_dir()
    env_path = _ensure_user_env(app_dir)
    os.environ["MASHUP_ENV_FILE"] = str(env_path)
    os.environ.setdefault("MASHUP_DESKTOP", "1")
    sessions = app_dir / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MASHUP_SESSIONS_ROOT", str(sessions))
    os.environ.setdefault("MASHUP_PORT", str(port))

    # Load user env before importing main (main also load_dotenv's).
    from dotenv import load_dotenv

    load_dotenv(env_path, override=False)
    # Local convenience: when running from source, fill blanks from repo .env.
    repo_env = Path(__file__).resolve().parent / ".env"
    if repo_env.is_file() and not getattr(sys, "frozen", False):
        load_dotenv(repo_env, override=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AI Song Mashup desktop server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("MASHUP_PORT", "8765")))
    args = parser.parse_args(argv)

    _prepare_env(args.port)
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("desktop_server")
    log.info("Config: %s", os.environ.get("MASHUP_ENV_FILE"))
    log.info("Sessions: %s", os.environ.get("MASHUP_SESSIONS_ROOT"))
    log.info("Listening on http://%s:%s", args.host, args.port)

    import uvicorn

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
