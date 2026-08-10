# AI Song Mashup POC

<img width="753" height="664" alt="Screenshot 2026-08-10 at 3 41 01 AM" src="https://github.com/user-attachments/assets/4814f631-0bd4-4dac-87f5-1ff132a6e6ed" />
<img width="711" height="664" alt="Screenshot 2026-08-10 at 7 25 25 AM" src="https://github.com/user-attachments/assets/587e95b6-e510-45b6-9f87-4c66ec564825" />


Local mashup app: upload **Song A** + **Song B**, separate stems with **Demucs**, choose **structure mode** (allin1 / LLM / DSP), let an **LLM** plan the creative arc, mix with **librosa / pydub**, then edit in a beginner **Section editor** grid.

Includes:

- **FastAPI** backend (`POST /api/mashup`)
- Drag-and-drop **web UI** (+ in-page player) at `/`
- Fully **local** stem separation (no paid separation API)
- **Section editor** — Song A/B × stem rows × mashup section columns (GarageBand-simple)

---

## Architecture

```text
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  static UI  │────▶│   main.py    │────▶│ session / MP3   │
│  (app.js)   │◀────│  FastAPI     │◀────│ studio.json     │
└─────────────┘     └──────┬───────┘     └─────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   demucs.py      structure resolvers     agent.py
   (WAV stems)    allin1 / LLM / DSP      (arrangement LLM)
         │                 │                 │
         └────────┬────────┴────────┬────────┘
                  ▼                 ▼
            dual_mix.py ◀──── audio.py / structure.py
            (+ studio_mix.py for Section editor renders)
```

### Who owns what

| Concern | Owner | Notes |
|---------|--------|--------|
| Stem separation | **Demucs** (`demucs.py`) | WAV four-stems; shared with allin1 demix cache |
| Section boundaries | **structure_mode** | `allin1` (best), `llm` (fast/rough), `dsp` (heuristic) |
| Creative arc | **LLM** (`agent.py`) | Full Song A spine preferred; A intro/outro bookends |
| Key / stretch / mix | **librosa** / **pydub** | Pitch meeting + BPM stretch |
| Section editor grid | **studio_mix.py** + UI | Seeded from auto mashup; user edits sources |

Song A is the **instrumental bed**. Song B can contribute selective overlays and vocals.

---

## How it works (pipeline)

1. **Upload** — multipart `song_a` / `song_b` (+ optional `structure_mode`).
2. **Demucs once** — WAV stems into a shared demix root.
3. **Structure** — depending on `structure_mode`:
   - `allin1` — reuses shared demix (skips second Demucs when cache hits)
   - `llm` — title/duration form guess, snapped to file length
   - `dsp` — `detect_sections` only
4. **Arrangement** — `decide_arrangement` (or full-A-spine fallback). Timeline may be up to 48 segments so mashups stay near Song A length.
5. **Mix** — stretch/pitch → section clips → `mashup.mp3` + session stems + `studio.json`.
6. **Section editor** — grid under the auto player; Play edit / Download edit via studio APIs.

Restore a finished session in the UI with `/?session=<id>`.

---

## Project layout

```text
ai-mashup-poc/
├── main.py                  # FastAPI routes + mashup orchestration
├── requirements.txt
├── .env.example
├── scripts/smoke_allin1.py
├── services/
│   ├── demucs.py            # Local Demucs (WAV four-stems)
│   ├── allin1_structure.py  # allin1 resolver (+ demix_dir share)
│   ├── form_analysis.py     # LLM form + DSP resolvers
│   ├── structure.py         # Section model + snap/detect helpers
│   ├── agent.py             # Arrangement LLM + full-A fallback
│   ├── dual_mix.py          # Mix + session persistence
│   ├── studio_mix.py        # Section editor seed/render
│   ├── audio.py / mashability.py / vad.py
└── static/                  # UI + Section editor grid
```

---

## Dependencies

### System

| Dependency | Role |
|------------|------|
| **Python 3.10+** | Runtime |
| **ffmpeg** | Decode/export; WAV for allin1 |
| **cmake** / **ninja** (for allin1) | Build `natten` |
| **rubberband** (optional) | Better stretch |

### Python packages

See `requirements.txt`. Key pins: `torch==2.2.2`, `natten==0.15.1`, `allin1==1.1.0` for the accurate structure path.

If `natten` / `allin1` cannot build in your environment, use `structure_mode=llm` or `dsp` — Demucs + mix still work.

### macOS allin1 install (order matters)

```bash
pip install -r requirements.txt
pip install Cython "setuptools>=70,<81"
pip install --no-build-isolation "git+https://github.com/CPJKU/madmom.git"
pip install torch==2.2.2 torchaudio==2.2.2
pip uninstall -y natten cmake
pip install --no-build-isolation --no-cache-dir natten==0.15.1
pip install allin1==1.1.0
PYTHONPATH=. python scripts/smoke_allin1.py
```

Notes:

- Prefer WAV uploads (or ffmpeg convert).
- First allin1 run downloads Harmonix weights (`HF_TOKEN` helps rate limits).
- Shared Demucs WAV demix skips allin1’s second demix when the cache is prefilled.
- Metadata records `structure_source` / `structure_mode`.

---

## API keys / LLM provider

| Variable | Required? | Purpose |
|----------|-----------|---------|
| `LLM_PROVIDER` | Optional | `gemini` (default) or `openai` |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` | For LLM | Arrangement (+ LLM structure mode) |
| `HF_TOKEN` | Optional | Hugging Face auth for Harmonix downloads (`HF_TOKEM` typo alias accepted) |

Without an LLM key, a deterministic **full Song A spine** fallback arrangement is used.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Complete allin1 block above if you want structure_mode=allin1
cp .env.example .env
```

**Do not commit `.env`.**

---

## Run the app

```bash
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

| URL | What |
|-----|------|
| http://127.0.0.1:8000/ | Web UI |
| http://127.0.0.1:8000/?session=&lt;id&gt; | Restore mashup + Section editor |
| http://127.0.0.1:8000/docs | OpenAPI |
| http://127.0.0.1:8000/health | Health |

1. Drop **Song A** and **Song B**.
2. Pick **Structure** mode (default allin1).
3. Click **Mashup**.
4. Use the auto player / Download mashup.mp3.
5. Edit in **Section editor** (grid), then **Play edit** / **Download edit**.

```bash
curl -X POST http://127.0.0.1:8000/api/mashup \
  -F "song_a=@/path/to/song_a.mp3" \
  -F "song_b=@/path/to/song_b.mp3" \
  -F "structure_mode=llm" \
  --output mashup.mp3
```

### Session APIs

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/mashup/sessions/{id}` | Metadata |
| GET | `/api/mashup/sessions/{id}/mashup` | Auto mashup MP3 |
| GET/PUT | `/api/mashup/sessions/{id}/studio` | Section editor grid |
| GET | `/api/mashup/sessions/{id}/clips/{a\|b}/{stem}?section=` | Audition source clip |
| POST | `/api/mashup/sessions/{id}/studio/preview` | Render edit preview |
| POST | `/api/mashup/sessions/{id}/studio/render` | Download `mashup-edit.mp3` |

---

## Mashup length

Earlier builds capped the director timeline at **8** segments, which made mashups much shorter than Song A. The cap is raised (48) and the fallback arrangement now emits **one action per Song A section** so wall-clock length tracks stretched Song A (BPM stretch still changes duration).

---

## Performance notes

- Demucs once per song (WAV); allin1 reuses that demix when `structure_mode=allin1`.
- Harmonix analyze on CPU is still slow (minutes/song) — use `llm`/`dsp` for a fast escape hatch.
- Session dirs under `/tmp/mashup_sessions` keep stems + `studio.json` for editing.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `demucs CLI not found` | Activate `.venv`; `pip install demucs` |
| allin1 / NATTEN build fails | Use `structure_mode=llm` or `dsp`; or rebuild natten on a machine with a working C++ toolchain |
| Short mashup vs Song A | Ensure you’re on this build (full-A spine fallback + raised timeline cap) |
| HF download rate limits | Set `HF_TOKEN` in `.env` |

---

## Security / privacy

- Secrets only in `.env` (gitignored).
- Audio stays local for Demucs / allin1 / mix.
- Arrangement LLM receives section summaries, not raw audio.
- POC only: no auth / queue / production hardening.

---

## License / attribution

POC code is provided as-is for experimentation.

Demucs, allin1 / Harmonix, and NATTEN have their own licenses. Use only audio you have rights to mash up.
