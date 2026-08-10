# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Windows x64 mashup-server sidecar (no allin1/natten).

Run from repo root on Windows:
  pyinstaller mashup-server.spec --noconfirm
"""

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None

datas = [
    ("static", "static"),
    (".env.example", "."),
]

hiddenimports = [
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "main",
    "services.agent",
    "services.audio",
    "services.demucs",
    "services.dual_mix",
    "services.form_analysis",
    "services.mashability",
    "services.studio_mix",
    "services.structure",
    "services.vad",
    "services.allin1_structure",
]

binaries = []

for pkg in ("demucs", "torch", "torchaudio", "librosa", "sklearn", "scipy", "resampy"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        try:
            hiddenimports += collect_submodules(pkg)
        except Exception:
            pass

try:
    datas += collect_data_files("librosa")
except Exception:
    pass

a = Analysis(
    ["desktop_server.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "allin1",
        "natten",
        "madmom",
        "tkinter",
        "matplotlib",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="mashup-server",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="mashup-server",
)
