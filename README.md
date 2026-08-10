# AI Song Mashup POC

An overnight proof-of-concept that takes **two songs**, separates them into stems with **local Demucs**, uses an **LLM** to pick a mixing strategy, **time-stretches** the vocals to match BPM, and returns a mixed **MP3 mashup**.

Includes:

- A **FastAPI** backend (`POST /api/mashup`)
- A simple **drag-and-drop web UI** at `/`
- Fully **local stem separation** (no Replicate / paid separation API)

---

## How it works

```text
Song A + Song B
      │
      ▼
┌─────────────────────┐
│ 1. Save uploads     │  tempfile under /tmp
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 2. Demucs (local)   │  vocals + no_vocals for each song
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 3. Librosa BPM      │  estimate tempo for A and B
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 4. LLM decision     │  Gemini or OpenAI → MashupDecision
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 5. Time-stretch     │  align vocals to target BPM (librosa)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 6. Key match        │  chroma key detect + pitch_shift vocals
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 7. Mix + export     │  pydub overlay → mashup.mp3
└─────────────────────┘
```

### Pipeline details

1. **Upload** — The API accepts two multipart files: `song_a` and `song_b` (mp3, wav, flac, m4a, etc.).
2. **Stem separation** — `services/demucs.py` shells out to the Demucs CLI:

   ```bash
   demucs -n htdemucs --two-stems vocals --mp3 -o {output_dir} {input_audio}
   ```

   That produces:

   ```text
   {output_dir}/htdemucs/{song_name}/vocals.mp3
   {output_dir}/htdemucs/{song_name}/no_vocals.mp3
   ```

   `--mp3` uses `lameenc` for export (avoids a known `torchaudio` / `torchcodec` WAV save issue on newer PyTorch stacks).

3. **BPM detection** — `services/audio.py` uses `librosa.beat.beat_track` on each original upload.
4. **Mixing strategy** — `services/agent.py` calls **Gemini or OpenAI** (see `LLM_PROVIDER`) with a Pydantic structured output (`MashupDecision`):

   | Field | Meaning |
   |-------|---------|
   | `vocal_source` | `"song_a"` or `"song_b"` |
   | `instrumental_source` | the other song |
   | `target_bpm` | BPM to align to (usually the instrumental’s) |
   | `stretch_factor` | `target_bpm / vocal_source_bpm` |

   If the selected provider fails (or its API key is missing), a deterministic fallback is used: **A vocals + B instrumental**, stretched to B’s BPM.

5. **Time-stretch** — Vocals are stretched with `librosa.effects.time_stretch` (skipped if the factor is ~1.0).
6. **Key match** — Detect keys with chroma STFT profiles, then `pitch_shift` vocals toward the instrumental root.
7. **Mix** — `pydub` overlays vocals on the instrumental and exports `mashup.mp3` (requires **ffmpeg**).

Demucs runs in a worker thread via `asyncio.to_thread(...)` so the FastAPI event loop is not blocked during the long separation step.

---

## Project layout

```text
ai-mashup-poc/
├── main.py                 # FastAPI app, UI mount, /api/mashup pipeline
├── requirements.txt        # Python dependencies
├── .env.example            # Env var template (no secrets)
├── .gitignore
├── services/
│   ├── demucs.py           # Local Demucs CLI stem separation
│   ├── audio.py            # BPM, time-stretch, mix
│   └── agent.py            # Gemini/OpenAI MashupDecision + fallback
└── static/
    ├── index.html          # Drag-and-drop UI
    ├── styles.css
    └── app.js              # Upload → progress → download
```

---

## Requirements

### System

| Dependency | Why |
|------------|-----|
| **Python 3.10+** (3.11 recommended) | Runtime |
| **ffmpeg** | `pydub` MP3 decode/export |
| Disk + RAM | Demucs downloads ~80MB model weights on first run; separation is CPU/GPU heavy |

Install ffmpeg (macOS):

```bash
brew install ffmpeg
```

### API keys / LLM provider

| Variable | Required? | Purpose |
|----------|-----------|---------|
| `LLM_PROVIDER` | Optional | `gemini` (default) or `openai` |
| `GEMINI_API_KEY` | If using Gemini | Google AI Studio / Gemini API key |
| `GEMINI_MODEL` | Optional | Defaults to `gemini-flash-latest` |
| `OPENAI_API_KEY` | If using OpenAI | OpenAI API key |
| `OPENAI_MODEL` | Optional | Defaults to `gpt-4o-mini` |

Stem separation is **local and free**. Without a working LLM key for the selected provider, the app still works using the A-vocals / B-instrumental fallback.

Example `.env`:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-flash-latest
OPENAI_API_KEY=sk-your_openai_key_here
OPENAI_MODEL=gpt-4o-mini
```

Switch providers by changing `LLM_PROVIDER` and restarting uvicorn (or relying on `--reload` after env is loaded — restart is safer for env changes).

---

## Setup

```bash
# 1. Clone
git clone https://github.com/komeslik/ai-mashup-poc.git
cd ai-mashup-poc

# 2. Virtualenv
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install Python deps (pulls torch + demucs; can take a few minutes)
pip install -r requirements.txt

# 4. Env file
cp .env.example .env
# Edit .env — set LLM_PROVIDER and the matching API key
#   LLM_PROVIDER=gemini  + GEMINI_API_KEY=...
#   LLM_PROVIDER=openai  + OPENAI_API_KEY=...
```

**Do not commit `.env`.** It is gitignored. Only `.env.example` (placeholders) belongs in the repo.

---

## Run the app

```bash
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

Then open:

| URL | What |
|-----|------|
| http://127.0.0.1:8000/ | Web UI (drag-drop two tracks → Mashup → download) |
| http://127.0.0.1:8000/docs | Swagger / OpenAPI |
| http://127.0.0.1:8000/health | Health check (`{"status":"ok"}`) |

### Using the UI

1. Drop **Song A** and **Song B** into the two zones (or click to browse).
2. Click **Mashup**.
3. Wait — Demucs runs **twice** (once per song). On CPU this often takes several minutes; first ever run also downloads model weights.
4. When finished, the progress bar hides and a **Download mashup.mp3** button appears.

Progress text in the UI is approximate (the API is one long request). The elapsed timer is real.

### Using curl

```bash
curl -X POST http://127.0.0.1:8000/api/mashup \
  -F "song_a=@/path/to/song_a.mp3" \
  -F "song_b=@/path/to/song_b.mp3" \
  --output mashup.mp3
```

### Using Swagger

1. Open http://127.0.0.1:8000/docs  
2. Expand `POST /api/mashup`  
3. Upload two files and execute  
4. Download the response body as an MP3  

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
| `song_a` | file | First track |
| `song_b` | file | Second track |

**Success:** `200` with `audio/mpeg` body (`mashup.mp3`).

**Common errors:**

| Status | Meaning |
|--------|---------|
| `400` | Bad upload / BPM detection failure |
| `500` | Demucs CLI failure, mix/ffmpeg failure, or unexpected error |
| `502` | Upstream LLM failure that wasn’t covered by fallback (rare) |

---

## Configuration

Create `.env` from the example:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-your_openai_key_here
OPENAI_MODEL=gpt-4o-mini
```

---

## Performance notes

- **First Demucs run** downloads the `htdemucs` checkpoint (~80MB) into Demucs’ cache.
- **Each mashup** separates two full songs locally. Short clips are much faster for testing.
- On Apple Silicon, PyTorch uses CPU/MPS depending on your torch build; expect multi-minute runs for full-length tracks.
- Temporary work files live under `/tmp` and are cleaned up after the response (final MP3 is deleted via a background task after download).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `demucs CLI not found` | Activate `.venv` and reinstall: `pip install demucs` |
| `ffmpeg` / MP3 export errors | `brew install ffmpeg` and ensure it’s on `PATH` |
| OpenAI errors in logs | Check `OPENAI_API_KEY`; app should still mashup via fallback |
| UI stuck / long wait | Normal during Demucs; watch the uvicorn terminal for `Separating stems…` logs |
| Import / torchcodec WAV errors | This project uses Demucs `--mp3` to avoid that path |

---

## Tech stack

- **FastAPI** + **Uvicorn** — HTTP API and static UI
- **Demucs 4** (`htdemucs`) — local music source separation
- **Librosa** + **NumPy** + **SoundFile** — BPM and time-stretch
- **pydub** + **ffmpeg** — mixing and MP3 export
- **OpenAI** + **Pydantic** — structured mashup decision
- **python-dotenv** — local config

---

## Security / privacy

- Secrets belong only in `.env` (gitignored).
- Uploaded audio is processed locally for stem separation.
- OpenAI receives **BPM numbers only** (not the audio files) when deciding strategy.
- This is a **POC**: no auth, no job queue, no production hardening.

---

## License / attribution

POC code is provided as-is for experimentation.

Demucs is from Meta/FAIR research; respect Demucs’ and model license terms when redistributing or commercializing. Use only audio you have rights to mash up.
