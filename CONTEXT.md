# Irodori-TTS（いろとりTTS）Mac ローカル運用メモ

## これは何か
Aratako 氏の日本語特化ローカル TTS。Flow Matching ベース、絵文字で感情制御、
参照音声でゼロショット声クローン、キャプション文で声質デザイン（VoiceDesign）。
本家: https://github.com/Aratako/Irodori-TTS

## この Mac での構成
- 置き場所: `~/git/Irodori-TTS`。remote は **origin = dezigozi/Irodori-TTS（自分のfork）／upstream = Aratako/Irodori-TTS（本家）**。
  本家の更新を取り込むときは `git fetch upstream && git merge upstream/main`
- `refs/`（本人の声）と `outputs/` は `.gitignore` で除外。**fork は public なので絶対に声を commit しない**
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

## いろとりスタジオ（app/）— 声のデータベース作成＋原稿読み上げ保存アプリ
Tauri v2 のデスクトップアプリ（`app/`）。Vite 無し・`app/src/index.html` 1枚・`frontendDist: ../src` の
かべちゃん方式なので **dev ポート不要**。バックエンドは上の常駐サーバ（:3952）をそのまま使い、
アプリ側の Rust は「サーバ起動・設定保存・Finder/ファイル操作」だけ。話者操作や合成はフロントの fetch が直接叩く。

- 上段「声のデータベース」: 声の元ファイル（m4a/wav/mp3/mp4…）をドロップ → 名前を入れる → `POST /speakers`
  → refs/ に 10秒×最大3本のクリップ＋latent ができて `speakers.json` に登録される（3秒前後）
- 下段「原稿 → 音声」: 話者・抑揚（flat/natural/rich）・演技メモ（自由キャプション）・速さ → `POST /render`
  → ジョブをポーリングして進捗バー → **48kHz/mono/16bit WAV** を保存先（既定 `~/Music/いろとりスタジオ/`）に保存、そのまま再生
- 設定は `~/Library/Application Support/com.takataka.irodoristudio/settings.json`、サーバログは同 `server.log`
- ビルド: `cd app && npm install && npx tauri build` → `app/src-tauri/target/release/bundle/macos/いろとりスタジオ.app` を /Applications へ
- アイコン: `app/app-icon.svg` → `rsvg-convert` で `app-icon.png` → `npx tauri icon app-icon.png`

### server.py に足した API（0.2.0）
| API | 用途 |
|---|---|
| `POST /speakers` `{name, source_path, clip_seconds=10, max_clips=3}` | ffprobe→ffmpeg で `refs/<id>_full.wav`（mono/pcm_s24le/元SR維持、48k超は48kへ）→ 均等配置で `refs/<id>_a..c.wav`（無音気味なら5秒ずらし）→ latent を焼いて `speakers.json` に原子的に追記。id は `spk_<epoch>` |
| `DELETE /speakers/<id>` | speakers.json から外し、`refs/<id>_*.wav` と latent を削除。`takataka` は保護、最後の1人も消せない |
| `POST /synthesis` 追加引数 | `caption`（自由文、expression より優先）／`sample_rate`（22050 既定・48000）。既存引数は変更なし |
| `POST /render` `{speaker, text, expression|caption, rate, num_steps?, out_path, pause_ms=350}` | 原稿を `。！？!?改行` で切って ≤110文字にまとめ、チャンクごとに合成→無音を挟んで連結→48kHz WAV を一時ファイル経由で保存。ジョブIDを返す |
| `GET /jobs/<id>` / `POST /jobs/<id>/cancel` | 進捗 `{status, done, total, out_path, seconds, error}`／チャンク境界で中断 |
- CORS（`Access-Control-Allow-Origin: *` と OPTIONS）を返す。Tauri の webview から fetch するため
- `/version` の `sample_rate` は econte 互換のため 22050 のまま。`native_sample_rate: 48000` を追加
- `GET /speakers` は `{id,name,clips}` に `clip_paths/source/created_at/protected` を**足しただけ**（econte の読み方は壊れない）
- 実測: 話者追加 3秒（2分42秒の m4a）／レンダーは1チャンク 15秒前後（2チャンク=15.6秒の音声で約35秒）
