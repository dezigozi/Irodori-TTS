#!/bin/bash
# econte など外部アプリ向けの常駐HTTPサーバ
# ポート 3952 は ~/.claude/PORTS.md で irodori-tts に予約済み
set -euo pipefail
cd "$(dirname "$0")"
exec uv run --no-sync python server.py --host 127.0.0.1 --port 3952 "$@"
