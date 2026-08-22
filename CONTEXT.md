# Irodori-TTS（いろとりTTS）Mac ローカル運用メモ

## これは何か
Aratako 氏の日本語特化ローカル TTS。Flow Matching ベース、絵文字で感情制御、
参照音声でゼロショット声クローン、キャプション文で声質デザイン（VoiceDesign）。
本家: https://github.com/Aratako/Irodori-TTS

## この Mac での構成
- 置き場所: `~/git/Irodori-TTS`（本家の shallow clone）
- 環境: `uv sync --extra cpu`（macOS では通常の PyPI PyTorch wheel が入り MPS が使える）
- デバイス: `mps` が自動で既定になる。精度は MPS/CPU では fp32 のみ（bf16 は CUDA/XPU 専用）
- モデル: `Aratako/Irodori-TTS-v4.1-Small`（HF キャッシュ `~/.cache/huggingface`、約 7.7GB）
- ポート: 3950 = 標準WebUI / 3951 = VoiceDesign WebUI（`~/.claude/PORTS.md` に予約済み）

## 追加した自前スクリプト（本家には無い）
| ファイル | 用途 |
|---|---|
| `start_webui.sh` | 標準WebUI を :3950 で起動（参照音声クローン + Speaker Inversion） |
| `start_webui_voicedesign.sh` | VoiceDesign WebUI を :3951 で起動（キャプションで声質指定） |
| `say.sh` | たかたかの声で一発生成 `./say.sh "文章" [出力.wav] [キャプション]` |
| `server.py` / `start_server.sh` | 外部アプリ向けの常駐HTTPサーバを :3952 で起動 |
| `speakers.json` | サーバが使う話者の定義（id → 表示名 + 参照クリップ） |

## たかたかの参照音声
- 元データ: `~/Downloads/自分の音声.m4a`（2分42秒 / **ALAC ロスレス / 48kHz mono / 24bit**）
- `refs/takataka_src.m4a` にコピー → `refs/takataka_full.wav`（48kHz mono / pcm_s24le）へ変換
- そこから 10 秒 × 3 本を切り出し: `refs/takataka_a.wav`（5秒〜）/ `_b.wav`（60秒〜）/ `_c.wav`（120秒〜）
- v4-Small は「短いクリップを複数」が学習時の構成に合う。合計30秒で話者類似度の伸びはほぼ頭打ち（上限120秒）
- **48kHz のまま渡すこと**。コーデックが内部で1回だけリサンプルするので、こちらで 44.1kHz に落とすと変換が二重になる
- ALAC を ffmpeg でデコードすると `invalid samples per frame: 0` が1行出るが、末尾の空フレームで無害
  （Peak -2.4dB / Flat factor 0 / サンプル数 7,819,264 = 162.9秒 と健全性は確認済み）
- 旧・低品質版（16kHz/24kbps mp3 由来）は A/B 比較用に `refs/old_mp3/` に退避

### 参照音声の品質差（実測 A/B）
同じ文章・同じシード（4502553081600545860）で生成して比較した結果、
生成音の 8kHz 以上のエネルギーが **-59.1dB → -51.7dB（+7.5dB）**。
参照がロスレスになると高域のディテール（サ行・息づかい）がはっきり乗る。
- 旧: `outputs/takataka_clone.wav` / 新: `outputs/takataka_clone_lossless.wav`

## 罠・注意点
- **参照音声の質が上限を決める**。実際に 16kHz/24kbps mp3 → 48kHz/24bit ALAC に差し替えたら
  生成音の高域が +7.5dB 改善した。ここをケチると何をチューニングしても頭打ちになる
- 生成音声には **SilentCipher の電子透かしが自動で入る**（依存とモデルが揃っている場合）
- 初回実行はモデル DL で 6〜8 分かかる。2回目以降はキャッシュから即ロード
- 倫理制約: 本人の同意なき第三者の声の模倣・ディープフェイク用途は禁止（本家 LICENSE / README 参照）
- `--seconds` は基本渡さない。v4-Small は尺を自動予測する。伸ばしたいときは `--duration-scale 1.2` などで倍率指定

## 実測（M5 / 32GB / MPS）
8秒の音声を生成して合計 12〜17 秒（モデルロード後）。内訳は sample_rf が支配的。
`--num-steps 6 --t-schedule-mode sway --sway-coeff -1.0` で高速化できる（品質とのトレードオフ）

## 常駐HTTPサーバ（server.py）— econte 連携用
本家には Gradio UI と CLI しか無いので、外部アプリから叩くための最小サーバを自前で足した。

```bash
./start_server.sh              # :3952 で起動
curl http://127.0.0.1:3952/version
curl http://127.0.0.1:3952/speakers
curl -X POST http://127.0.0.1:3952/synthesis -H "Content-Type: application/json" \
  -d '{"speaker":"takataka","text":"こんにちは","expression":"natural","rate":180}' -o out.wav
```

- **速さのための2点**: ①モデルを常駐させる（毎回のロード約6秒が消える）
  ②**参照音声の latent を起動時に焼いて `refs/.latents/` に置く**（毎回の参照エンコード約4秒が消える）。
  結果 **1行9秒前後**（`--num-steps` を既定40→32に落としてある）
- **出力は 22050Hz / モノラル / 16bit の WAV 固定**。econte のナレーション連結パイプラインが
  この形式しか受け取らないので、サーバ側で揃えて返す（呼び出し側で変換を挟まずに済む）
- `expression` は `flat` / `natural` / `rich` の3段階。Irodori は stability のような数値ノブを持たないので、
  **日本語の演技キャプション**に写している（`EXPRESSION_CAPTIONS`）
- `rate` は say の180を等速とみなして `duration_scale = 180/rate` に写す（0.6〜1.8でclamp）
- **シードは本文ハッシュから作る**ので、同じ文章は毎回同じ音になる。
  呼び出し側の行キャッシュを消して作り直したときに、その行だけ音が変わる事故を防ぐため。
  **別テイクが欲しいときは `take`（0以外の整数）を渡す**とシードに混ざって違う読みになる（econteの🔁録り直しが使う）
- 合成は `threading.Lock` で1件ずつ直列に流す（MPS上のモデルを同時に叩かせない）
- 参照wavを差し替えたら latent は**自動で焼き直す**（パス・サイズ・更新時刻の指紋で判定）

### 話者を増やすとき
`speakers.json` に足して、参照クリップを `refs/` に置いてサーバを再起動するだけ。
```json
{
  "takataka": { "name": "たかたか（本人）", "clips": ["refs/takataka_a.wav", "refs/takataka_b.wav", "refs/takataka_c.wav"] },
  "誰か":     { "name": "だれかの声",       "clips": ["refs/dareka_a.wav"] }
}
```
