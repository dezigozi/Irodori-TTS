#!/bin/bash
# Irodori-TTS 標準WebUI（参照音声クローン + Speaker Inversion）
# ポート 3950 は ~/.claude/PORTS.md で irodori-tts に予約済み
set -euo pipefail
cd "$(dirname "$0")"
exec uv run --no-sync python gradio_app.py --server-name 127.0.0.1 --server-port 3950
