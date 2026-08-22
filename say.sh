#!/bin/bash
# たかたかの声でしゃべらせる簡易CLI
#   ./say.sh "しゃべらせたい文章" [出力ファイル名] [キャプション]
# 例: ./say.sh "おはようさん、今日もええ天気やな😊" asa.wav "落ち着いた男性の声で、ゆっくり穏やかに"
set -euo pipefail
cd "$(dirname "$0")"

TEXT="${1:?しゃべらせる文章を渡してや}"
OUT="${2:-outputs/say_$(date +%Y%m%d_%H%M%S).wav}"
CAPTION="${3:-}"

REFS=(refs/takataka_a.wav refs/takataka_b.wav refs/takataka_c.wav)
for f in "${REFS[@]}"; do
  [[ -f "$f" ]] || { echo "参照音声がないで: $f" >&2; exit 1; }
done

ARGS=(
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small
  --text "$TEXT"
  --ref-wavs "${REFS[@]}"
  --output-wav "$OUT"
)
if [[ -n "$CAPTION" ]]; then
  ARGS+=(--caption "$CAPTION")
fi

uv run --no-sync python infer.py "${ARGS[@]}"
echo "できたで: $OUT"
afplay "$OUT" 2>/dev/null || true
