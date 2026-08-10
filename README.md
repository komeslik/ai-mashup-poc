# AI Song Mashup POC

Local mashup app: upload **Song A** + **Song B**, separate stems with **Demucs**, read real section timestamps with **allin1**, let an **LLM** plan the creative arc (not the clocks), then mix with **librosa / pydub** into an MP3.

Includes:

- **FastAPI** backend (`POST /api/mashup`)
- Drag-and-drop **web UI** (+ in-page player) at `/`
- Fully **local** stem separation and structure analysis (no paid separation API)

---

## Architecture

```text
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  static UI  │────▶│   main.py    │────▶│ session / MP3   │
│  (app.js)   │◀────│  FastAPI     │◀────│ metadata.json   │
└─────────────┘     └──────┬───────┘     └─────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   demucs.py      allin1_structure.py    agent.py
   (stems)        (sections+BPM)         (arrangement LLM)
         │                 │                 │
         └────────┬────────┴────────┬────────┘
                  ▼                 ▼
            dual_mix.py ◀──── audio.py / structure.py
            (A bed + B overlays, stretch, key, export)
```

### Who owns what

| Concern | Owner | Notes |
|---------|--------|--------|
| Stem separation (vocals/drums/bass/other) | **Demucs** (`demucs.py`) | Local `htdemucs`; runs on originals |
| Section boundaries + labels + BPM | **allin1** (`allin1_structure.py`) | Real audio timestamps; DSP fallback if allin1 fails |
| Beats / downbeats (when available) | **allin1** | Stored in session metadata |
| Creative arc (A/B vocals, overlays, hooks) | **LLM** (`agent.py`) | Gemini or OpenAI; never invents section clocks |
| Song A intro / outro bookends | **agent.py** post-validate | Always forced |
| Key meeting, pitch shift, time-stretch | **librosa** (+ optional rubberband) | allin1 does not do key/pitch |
| Mashability / chroma / rhythm scores | **librosa** (`mashability.py`) | Used when ranking / scoring |
| Vocal activity mute / anti-bleed | **vad.py** + mix rules | Hard mute inactive leads |
| Final concatenate + MP3 | **pydub** + **ffmpeg** | Adaptive crossfades |

Song A is the **instrumental bed**. Song B can contribute selective overlays (drums/bass) and vocals per the director timeline.

---

## How it works (pipeline)

```text
Song A + Song B
      │
      ▼
┌─────────────────────┐
│ 1. Save uploads     │  tempfile under /tmp
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 2. Demucs (local)   │  full stems (vocals/drums/bass/other)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 3. allin1 structure │  real intro/verse/chorus timestamps + BPM
│    (librosa fallback)│
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 4. LLM arrangement  │  Gemini/OpenAI creative arc (not timestamps)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 5. Key + stretch    │  librosa chroma / pitch / time-stretch
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 6. Mix + export     │  A bed + selective B overlays → mashup.mp3
└─────────────────────┘
```

1. **Upload** — multipart `song_a` / `song_b` (mp3, wav, flac, m4a, …).
2. **Stem separation** — `services/demucs.py` → full stems per song.
3. **Structure + BPM** — `allin1` on each **original** (WAV via ffmpeg when needed). Prefer allin1 BPM; on failure → `detect_sections` and `structure_source: "dsp_fallback"`.
4. **Arrangement** — `decide_arrangement` builds the timeline; A intro/outro bookends injected. Bassline mode uses `decide_mashup_strategy` instead.
5. **DSP** — key meeting (`minimal_meeting_shifts`), pitch shift, time-stretch to target BPM.
6. **Mix** — section clips, selective B overlays, adaptive crossfades → `mashup.mp3`.

Heavy work runs in `asyncio.to_thread(...)` so the event loop stays responsive.

---

## Project layout

```text
ai-mashup-poc/
├── main.py                  # FastAPI routes + mashup orchestration
├── requirements.txt         # Pinned Python deps (see below)
├── .env.example
├── scripts/
│   └── smoke_allin1.py      # Verify true allin1 path (not DSP fallback)
├── services/
│   ├── demucs.py            # Local Demucs CLI stem separation
│   ├── allin1_structure.py  # allin1 → Section list + BPM/beats
│   ├── form_analysis.py     # Thin re-export of allin1 resolver
│   ├── structure.py         # Section model + DSP detect_sections
│   ├── agent.py             # Arrangement LLM + A bookends
│   ├── dual_mix.py          # Dual-vocal / bassline builders + metadata
│   ├── audio.py             # BPM (librosa), stretch, key, pitch helpers
│   ├── mashability.py       # Chroma / rhythm mashability scores
│   ├── vad.py               # Vocal activity helpers
│   └── library.py           # Optional local track library ranking
└── static/                  # UI: upload, progress, player, download
```

---

## Dependencies

### System

| Dependency | Role |
|------------|------|
| **Python 3.10+** (3.11 recommended) | Runtime |
| **ffmpeg** | Decode/export MP3; convert uploads to WAV for allin1 timing |
| **cmake** + **ninja** | Compile `natten` on macOS |
| **rubberband** (optional) | Higher-quality time-stretch via `pyrubberband` |

```bash
brew install ffmpeg cmake ninja
brew install rubberband   # optional
```

### Python packages (`requirements.txt`)

| Package | Role in this project |
|---------|----------------------|
| **fastapi** / **uvicorn** | HTTP API, static UI, `/api/mashup` |
| **python-multipart** | File uploads |
| **torch** / **torchaudio** (`2.2.2`) | Backend for Demucs + allin1/NATTEN (pin required on Mac) |
| **demucs** | Local source separation (vocals/drums/bass/other) |
| **allin1** | Music structure: segments, BPM, beats, downbeats |
| **natten** (`0.15.1`) | Neighborhood attention kernels required by allin1 |
| **madmom** (GitHub install) | Audio frontend used inside allin1 |
| **Cython** / **setuptools&lt;81** | Build madmom; keep `pkg_resources` available |
| **librosa** / **numpy** / **scipy** / **soundfile** | BPM fallback, key, stretch, spectral features, I/O |
| **pydub** | Clip mix, crossfades, MP3 export (needs ffmpeg) |
| **pyloudnorm** | Loudness normalization helpers |
| **pyrubberband** | Optional rubberband stretch wrapper |
| **openai** / **google-genai** | Arrangement LLM providers |
| **pydantic** | Structured LLM / blueprint schemas |
| **python-dotenv** | Load `.env` |
| **httpx** / **requests** | HTTP clients used by LLM SDKs / helpers |

### What each major tool does *not* do

| Tool | Does **not** |
|------|----------------|
| allin1 | Key detection, pitch shift, creative arrangement |
| LLM | Invent section start/end timestamps (replaced by allin1) |
| Demucs | Structure labels / BPM (allin1 may demix again internally for its own features) |
| librosa | Primary structure when allin1 succeeds |

### macOS allin1 install (order matters)

Working Apple Silicon combo: **torch 2.2.2 + natten 0.15.1 + allin1 1.1.0**. Newer torch/`natten` 0.21 breaks allin1’s imports.

```bash
pip install -r requirements.txt

pip install Cython "setuptools>=70,<81"
pip install --no-build-isolation "git+https://github.com/CPJKU/madmom.git"

pip install torch==2.2.2 torchaudio==2.2.2
pip uninstall -y natten cmake    # drop broken pip cmake wheel if present
pip install --no-build-isolation --no-cache-dir natten==0.15.1
pip install allin1==1.1.0

PYTHONPATH=. python scripts/smoke_allin1.py   # expect: source allin1
```

Notes:

- Prefer **WAV** uploads (or let the app convert via ffmpeg) — MP3 decoder offsets can skew beats.
- First allin1 run downloads Harmonix weights from Hugging Face (`HF_TOKEN` helps if rate-limited).
- allin1 demixes into the job work dir; mix stems still come from our Demucs pass on the originals.
- If analyze fails → DSP fallback; session metadata records `structure_source`.

---

## API keys / LLM provider

| Variable | Required? | Purpose |
|----------|-----------|---------|
| `LLM_PROVIDER` | Optional | `gemini` (default) or `openai` |
| `GEMINI_API_KEY` | If Gemini | Google AI Studio / Gemini API key |
| `GEMINI_MODEL` | Optional | Default `gemini-flash-latest` |
| `OPENAI_API_KEY` | If OpenAI | OpenAI API key |
| `OPENAI_MODEL` | Optional | Default `gpt-4o-mini` |

Stem separation and allin1 are **local**. Without a working LLM key, a deterministic A-anchor arrangement fallback is used.

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-flash-latest
OPENAI_API_KEY=sk-your_openai_key_here
OPENAI_MODEL=gpt-4o-mini
```

---

## Setup

```bash
git clone https://github.com/komeslik/ai-mashup-poc.git
cd ai-mashup-poc

python3 -m venv .venv
source .venv/bin/activate

brew install ffmpeg cmake ninja
pip install -r requirements.txt
# Then complete the macOS allin1 block above (madmom + natten rebuild).

cp .env.example .env
# Edit .env — set LLM_PROVIDER and the matching API key
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
| http://127.0.0.1:8000/docs | Swagger / OpenAPI |
| http://127.0.0.1:8000/health | `{"status":"ok"}` |

1. Drop **Song A** and **Song B**.
2. Click **Mashup**.
3. Wait — Demucs ×2 plus allin1 (which demixes again). First run also downloads models.
4. Play in-page or **Download mashup.mp3**.

```bash
curl -X POST http://127.0.0.1:8000/api/mashup \
  -F "song_a=@/path/to/song_a.mp3" \
  -F "song_b=@/path/to/song_b.mp3" \
  --output mashup.mp3
```

---

## API reference

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /api/mashup`

**Content-Type:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `song_a` | file | Anchor track (instrumental bed) |
| `song_b` | file | Guest track (vocals / overlays) |

**Success:** `200` `audio/mpeg` (`mashup.mp3`). Session metadata (sections, `structure_source`, blueprint) is written under the session dir for the UI.

| Status | Meaning |
|--------|---------|
| `400` | Bad upload / BPM detection failure |
| `500` | Demucs / mix / ffmpeg failure |
| `502` | Strategy/arrangement failure |

---

## Performance notes

- First Demucs + allin1 runs download model weights (Demucs cache + Hugging Face Harmonix).
- Each mashup: Demucs twice + allin1 twice (allin1’s own demix). Short clips are much faster for testing.
- Torch pinned to **2.2.2** for allin1; expect multi-minute runs for full tracks on CPU/MPS.
- Temp work under `/tmp`; final MP3 cleaned up after download via background task.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `demucs CLI not found` | Activate `.venv`; `pip install demucs` |
| `ffmpeg` / MP3 errors | `brew install ffmpeg` |
| allin1 / NATTEN import errors | Keep `torch==2.2.2` + rebuild `natten==0.15.1` (see Dependencies) |
| `natten` cmake / build errors | `brew install cmake ninja`; `pip uninstall -y cmake` |
| `structure_source: dsp_fallback` | allin1 failed at runtime; check uvicorn logs; re-run smoke test |
| LLM errors in logs | Check API keys; arrangement fallback still mashups |
| Long UI wait | Expected (Demucs + allin1); watch terminal progress |

---

## Security / privacy

- Secrets only in `.env` (gitignored).
- Audio stays local for Demucs / allin1 / mix.
- The arrangement LLM receives **structured role maps / section summaries** (labels, energies, indices, BPMs) — not raw audio files.
- POC only: no auth, no job queue, no production hardening.

---

## License / attribution

POC code is provided as-is for experimentation.

Demucs (Meta/FAIR), allin1 / Harmonix models, and NATTEN have their own licenses — respect them when redistributing. Use only audio you have rights to mash up.
