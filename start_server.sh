#!/bin/bash
# econte など外部アプリ向けの常駐HTTPサーバ
# ポート 3952 は ~/.claude/PORTS.md で irodori-tts に予約済み・固定（別ポートへの繰り上げはしない）
set -euo pipefail
cd "$(dirname "$0")"

PORT=3952

# ポートガード: 3952が既に使われてたら中身を確認する
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  if curl -s -m 3 "http://127.0.0.1:$PORT/version" | grep -q '"engine": "irodori-tts"'; then
    echo "いろとりTTSサーバは既に :$PORT で動いてる。二重起動はしません" >&2
    exit 0
  fi
  echo "エラー: ポート $PORT を別のプロセスが使ってる。いろとりTTSは別ポートに逃げません。" >&2
  echo "占有プロセス:" >&2
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >&2
  exit 1
fi

exec uv run --no-sync python server.py --host 127.0.0.1 --port "$PORT" "$@"
