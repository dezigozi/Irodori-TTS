#!/bin/bash
# Irodori-TTS VoiceDesign WebUI（キャプションで声質を文章指定）
# ポート 3951 は ~/.claude/PORTS.md で irodori-tts に予約済み
set -euo pipefail
cd "$(dirname "$0")"
exec uv run --no-sync python gradio_app_voicedesign.py --server-name 127.0.0.1 --server-port 3951
