#!/bin/bash
# たかたかの声でしゃべらせる簡易CLI
#   ./say.sh "しゃべらせたい文章" [出力ファイル名] [キャプション] [半音]
# 例: ./say.sh "おはようさん、今日もええ天気やな😊" asa.wav "落ち着いた男性の声で、ゆっくり穏やかに"
# 半音を渡すと、速さはそのままで声の高さだけ上げる（7 で早回しっぽい高音。キャプション無しなら "" を渡す）
# 例: ./say.sh "こんにちは！" hi.wav "" 7
set -euo pipefail
cd "$(dirname "$0")"
# Finder やポータルから起動されたアプリの PATH は /usr/bin:/bin しか無く、uv（Homebrew）が見つからへん
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

TEXT="${1:?しゃべらせる文章を渡してや}"
OUT="${2:-outputs/say_$(date +%Y%m%d_%H%M%S).wav}"
CAPTION="${3:-}"
SEMITONES="${4:-0}"
[[ "$SEMITONES" =~ ^-?[0-9]+(\.[0-9]+)?$ ]] || { echo "半音は数字で渡してや: $SEMITONES" >&2; exit 1; }

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

if [[ "$SEMITONES" != "0" ]]; then
  # asetrate で高さと速さを両方変えてから、atempo で速さだけ元に戻す（atempo は 0.5〜2.0 しか受けへん＝±12半音まで）
  FACTOR=$(awk -v s="$SEMITONES" 'BEGIN { printf "%.6f", 2 ^ (s / 12) }')
  TEMPO=$(awk -v f="$FACTOR" 'BEGIN { printf "%.6f", 1 / f }')
  awk -v t="$TEMPO" 'BEGIN { exit !(t >= 0.5 && t <= 2.0) }' || { echo "半音は -12〜12 にしてや: $SEMITONES" >&2; exit 1; }
  SR=$(ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of csv=p=0 "$OUT")
  TMP="${OUT%.wav}.pitch_tmp.wav"
  ffmpeg -y -v error -i "$OUT" -af "asetrate=${SR}*${FACTOR},aresample=${SR},atempo=${TEMPO}" "$TMP"
  mv "$TMP" "$OUT"
fi
echo "できたで: $OUT"
afplay "$OUT" 2>/dev/null || true
