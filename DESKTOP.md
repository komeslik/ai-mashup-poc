# Desktop Windows app (Electron)

Keep using **localhost** for day-to-day work on your Mac:

```bash
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

This document covers the **additional** Windows desktop installer for sharing with a friend.

## What your friend gets

- `AI-Song-Mashup-Setup-0.3.0.exe` (NSIS installer, x64)
- Double-click → Electron window → local FastAPI backend on `127.0.0.1`
- Same UI (mashup + Section editor)
- Structure modes: **`llm`** and **`dsp`** (Demucs included). **`allin1` is not bundled** in the Windows build (too fragile/`natten`); the UI disables it automatically.

## API keys (friend setup)

On first run the app creates:

`%APPDATA%\ai-mashup-poc\.env`

Edit that file (Notepad) and set at least:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=their_or_your_shared_key
```

Optional: `HF_TOKEN=` (not required without allin1).

Restart the app after saving.

Sessions are stored under `%APPDATA%\ai-mashup-poc\sessions\`.

## SmartScreen (unsigned builds)

Windows may show “Windows protected your PC”:

1. Click **More info**
2. Click **Run anyway**

Code signing (Authenticode) removes this later; not required for a private share.

## Build the Windows installer (GitHub Actions — recommended)

You develop on Mac; CI builds the `.exe`:

1. Push this repo to GitHub.
2. Actions → **Build Windows Desktop** → **Run workflow** (or push changes under `desktop/`).
3. Download the artifact **AI-Song-Mashup-Windows**.
4. Send the `.exe` to your friend + the API key instructions above.

Suggested friend demo: pick **Structure → LLM form** for a faster first mashup.

## Build on a Windows machine (optional)

```powershell
# From repo root
.\scripts\build_sidecar_win.ps1
# Put ffmpeg.exe in tools\ffmpeg\ beforehand, or copy into desktop\resources\sidecar\
cd desktop
npm install
npm run dist:win
# Output: desktop\dist\AI-Song-Mashup-Setup-*.exe
```

## Dev the Electron shell on Mac (no Windows .exe)

Uses your local Python venv as the sidecar:

```bash
source .venv/bin/activate
cd desktop
npm install
npm run dev
```

## Phase 2 (later)

Mac M3 `.dmg` — same Electron + sidecar pattern; not required for friend share.

## Size / hardware notes

- Installer can be **1–3+ GB** (torch + Demucs).
- Friend’s PC should have **≥16 GB RAM** ideally; 8 GB may struggle.
- First mashup downloads Demucs weights into the user cache (needs network once).
