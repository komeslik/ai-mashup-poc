# Build Windows mashup-server sidecar into desktop/resources/sidecar/
# Run from repo root in PowerShell (Windows x64).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "Installing desktop Windows requirements..."
python -m pip install --upgrade pip
python -m pip install -r requirements-desktop-win.txt

Write-Host "Running PyInstaller..."
python -m PyInstaller mashup-server.spec --noconfirm --clean

$OutDir = Join-Path $Root "desktop\resources\sidecar"
if (Test-Path $OutDir) { Remove-Item -Recurse -Force $OutDir }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Copy-Item -Recurse -Force "dist\mashup-server\*" $OutDir

# Bundle ffmpeg if present in tools/ffmpeg/ or download hint
$FfmpegSrc = Join-Path $Root "tools\ffmpeg\ffmpeg.exe"
if (Test-Path $FfmpegSrc) {
  Copy-Item -Force $FfmpegSrc (Join-Path $OutDir "ffmpeg.exe")
  Write-Host "Bundled tools/ffmpeg/ffmpeg.exe"
} else {
  Write-Host "WARNING: tools/ffmpeg/ffmpeg.exe not found — CI should download ffmpeg into sidecar/"
}

Write-Host "Sidecar ready at $OutDir"
Get-ChildItem $OutDir | Select-Object -First 20
