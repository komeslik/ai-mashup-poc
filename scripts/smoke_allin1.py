"""Smoke-test allin1 structure analysis (true path, not DSP fallback)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from services.allin1_structure import analyze_structure


def main() -> None:
    sr = 44100
    t = np.linspace(0, 20, sr * 20, endpoint=False)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t)
    y = 0.2 * env * np.sin(2 * np.pi * 220 * t)
    stereo = np.stack([y, y], axis=1).astype(np.float32)

    work = Path(tempfile.mkdtemp(prefix="allin1_smoke_"))
    wav = work / "smoke.wav"
    sf.write(str(wav), stereo, sr)
    print("wrote", wav)

    bundle = analyze_structure(str(wav), work_dir=work / "cache")
    print("source", bundle.source)
    print("bpm", bundle.bpm)
    print("n_sections", len(bundle.sections))
    print("labels", [s.label for s in bundle.sections])
    print("beats", len(bundle.beats), "downbeats", len(bundle.downbeats))
    if bundle.source != "allin1":
        raise SystemExit(f"expected allin1, got {bundle.source}")
    print("SUCCESS")


if __name__ == "__main__":
    main()
