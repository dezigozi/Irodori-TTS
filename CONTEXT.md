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
- **⚠️ index.html を変えたら `touch src-tauri/build.rs && cargo clean -p irodori-studio --manifest-path src-tauri/Cargo.toml --release` してからビルドする。**
  `--release` を付けへんと `Removed 0 files` になって**クリーンされず、また古い HTML が焼かれる**（実際に踏んだ）。
  Vite 無しの `frontendDist: ../src` 構成では、tauri のインクリメンタルビルドが index.html の変更を検知せず、
  **Rust は再コンパイルされるのに古い HTML が埋め込まれたままの .app ができる**（2026-08-24 に実際にハマった）。
  WebView のキャッシュ（`~/Library/Caches|WebKit/com.takataka.irodoristudio`）を消しても直らんかったらこれ
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

## 🎨 声をデザインする（0.3.0）— 実在の人の声なしでバーチャル話者を作る

**目的**: econte（ぷれぜん君）の仮ナレで使う声を、実在の人の音声を1本も使わずに用意する。
たかたか本人の声だけやとナレーターの声色を選べへんかった。権利面の心配もゼロになる。

**しくみ**: v4.1-Small は caption（どんな声かの説明文）で条件づけできるので、`no_ref=True` にすると
参照音声なしで声が出る（本家の VoiceDesign）。それを **参照クリップを持つ普通の話者として登録** する。
登録さえしてまえば `_build_request` も econte 連携も既存のまま動く＝ **econte 側は改修ゼロ**。

```
caption（文章）→ no_ref で1本の長い音声を生成 → 切り分けて refs/<id>_a,b.wav
   → latent を焼く → speakers.json に登録 → econte の 🔄 で声リストに出る
```

### 決めたこと・理由
- **参照クリップは「1回の生成を切り分けて」作る**（文ごとに作り直さへん）。
  no_ref は text が変わると声も揺れるので、1本録りを ffmpeg で切れば声の同一性が保証される。
  `DESIGN_REF_TEXT`（約100文字）で実測 20.9秒 → 既存の `build_reference_clips` で 10秒×2本
- **seed を必ず呼び出し側に返す**（`X-Irodori-Seed` ヘッダ／`GET /speakers` の `seed`）。
  caption だけでは声は決まらず seed で別人になる＝ガチャなので、気に入った声を呼び戻す唯一の手段
- 話者 id は `vox_<epoch>`（ファイル由来の `spk_` と区別する）。`speakers.json` に caption と seed も残す
- `cfg_scale_speaker=0.0`（参照が無いので話者ガイダンスを効かせない）。本家 VoiceDesign UI と同じ扱い

### 罠
- **seed は19桁になる。JSON の数値で返したら JS の Number（2^53）で精度が落ちて別の声になる。**
  `GET /speakers` も `X-Irodori-Seed` も **文字列**で返し、UI 側も文字列のまま持つこと
- 同じ caption + 同じ seed の再現性は検証済み（2回生成して WAV が md5 完全一致）。
  seed を変えれば同じ caption でも別人になることも確認済み
- 生成音には silentcipher の電子透かしが乗る。その処理中に一瞬 44100Hz を通る（出力は48kHz）。
  デザインした声はその音を参照にするので、原理上「コピーのコピー」になる

### 話者ゼロでも起動する（0.3.1）
`speakers.json` と `refs/` は .gitignore 済みなので、clone 直後や新しいマシンでは話者が1人もいない。
昔はそこで `SystemExit` していたが、**声デザインがあれば参照音声なしで話者を作れる**ので起動は通すようにした。

- `_prepare_speakers`: 話者ゼロなら warning を出して起動を続ける
- `_build_request`: 話者ゼロなら「先に声を作ってください」と 400 で返す（`next(iter())` の StopIteration 対策）
- `delete_speaker`: 「最後の1話者は消せない」制約を外した（作り直せるようになったので）
- 実測: speakers.json と refs/ を退避した状態でサーバが起動し、`POST /design/speakers` で
  20.7秒／クリップ2本の話者を作れることを確認済み

## 📁 声のフォルダ分け（0.3.3 → 0.4.0）
声が10人を超えて一覧が探しにくくなったので、**1声=1フォルダ（グループ）**を足した。

**所属の情報源は声側**（`speakers.json` の各話者の `group`。空文字＝未分類）。
ただしそれだけやと**空のフォルダが作れへん**ので、0.4.0 で `groups.json`（フォルダ名の配列＝並び順）を足した。
- `merged_groups()` = groups.json の並び順 ＋ 声にだけ書いてあるフォルダを後ろに足す（**自己修復**）。
  groups.json を消しても、speakers.json を手で直しても、フォルダが消えたように見えへん
- `PATCH /speakers/<id>` は `rename_speaker` → `update_speaker` に一般化。
  **`name` / `group` のうち渡されたキーだけ**更新する（既存の名前だけのPATCHはそのまま動く）。
  知らんフォルダ名を渡されたら groups.json にも足す＝ドラッグで新フォルダへ直接移せる
- **フォルダを消しても声は消えへん**（未分類に戻るだけ）。改名は中の声の group もまとめて書き換える

### 足した API（0.4.0）
| API | 用途 |
|---|---|
| `GET /groups` | フォルダ名の一覧（並び順） |
| `POST /groups` `{name}` | 空フォルダを作る。重複は 400 |
| `PATCH /groups/<name>` `{name}` | 改名。中の声の `group` も全部まとめて付け替える |
| `DELETE /groups/<name>` | フォルダだけ消す。中の声は未分類に戻る |
| `POST /speakers/order` `{ids}` | 声の並び順を変える（0.4.1）。**speakers.json のキー順＝表示順**という既存の約束のまま並べ替える。ids は並べ替えたい範囲（フォルダ1個ぶん）だけでよく、**その ids が今占めてる位置に順に入れ直す**＝他フォルダの声は1ミリも動かへん。知らん id は黙って捨てる（一覧が古いクライアントから来ても壊れへん） |
| `--log-level DEBUG` | 起動オプション。リクエスト1本ずつがログに出る（どのAPIを叩かれてるかの確認用） |

### スタジオUI
- 一覧がフォルダ見出しで区切られる（未分類は最下段）。**空フォルダも枠だけ出す**＝ドラッグの受け皿になる
- 上に絞り込みチップ（選択は `settings.spkFilter`。**消えたフォルダを選んだままにせん**よう毎回検証して落とす）
- **フォルダ見出しの ⋯** に「新しいフォルダを作る／名前を変える／このフォルダを消す」。声カードの ⋯ にも「📁 グループを変える」（datalist で既存名を候補に出す＝ほぼ同名フォルダの乱立を防ぐ）
- **ドラッグでフォルダ間を移動**できる。掴んでる間（`dragging`）は 15秒ごとの自動更新で描き直さへん
  （描き直すと掴んでるカードが消えて操作が飛ぶ）
- ⚠️ **カードのドラッグは HTML5 drag&drop では動かへん**。`dragDropEnabled: true`（tauri.conf.json）が
  WebView の drag&drop を横取りしてまい、`dragstart` は起きても **`drop` が飛んでこーへん**
  （掴めるのに離しても何も起きひん、という症状。2026-08-27 に実際に踏んだ）。
  かというて false にすると声ファイルのドロップ（`tauri://drag-drop`。ffmpeg に渡す**実パス**が要るので
  HTML5 の方では代替できひん）が死ぬ。なので **カード移動だけ pointer イベントで自前実装**しとる:
  pointerdown → 5px 動いたらゴーストを作って追従 → `document.elementFromPoint` で `.spk-group` を拾う
  → pointerup で `dataset.group` へ PATCH。ゴーストは `pointer-events:none`（下の本物を拾わせるため）、
  直後の click は `justDragged` で握り潰す（ドラッグの余韻で選択が変わらんように）
- **フォルダ内の並べ替えも同じドラッグでやる**（0.4.1）。落ちる位置には紫の縦線（`.spk-drop-line`）を出すが、
  座標計算やなく**実DOMに線そのものを差し込む**（flex の並びがそのまま答えになる＝行またぎでもズレへん）。
  離したときに body の子を上から舐めて「線のところに自分を入れた id 配列」を作り、
  フォルダが変わってたら `PATCH /speakers/<id>`（group）→ 続けて `POST /speakers/order` を送る。
  同じフォルダで並びも変わってへんかったら通信せえへん
- 原稿側の話者セレクトも optgroup で束ねる

### ビルド後の焼き込み確認のコツ
Vite無し構成やから .app にHTMLが入ったかを確かめにくい（アセットは圧縮されてて `strings` に出えへん）。
**`--log-level DEBUG` でサーバを起動してアプリを開き、そのバージョンでしか叩かへんAPIがログに出るかを見る**のが確実
（0.4.0 なら `GET /groups`）。

### 足した API（0.3.0）
| API | 用途 |
|---|---|
| `POST /design` `{caption, text?, seed?, rate?, num_steps?, sample_rate?}` | 試し聞き。WAV を返し、使った種を `X-Irodori-Seed` ヘッダ（文字列）で返す。seed 省略で毎回ちがう声 |
| `POST /design/speakers` `{name, caption, seed, ref_text?, clip_seconds=10, max_clips=3}` | その声を話者として登録するジョブを開始。`GET /jobs/<id>` の `result` に話者が入る |
| `GET /version` | `voice_design: true/false`（チェックポイントが caption 条件づけ対応か）を追加 |
| `GET /speakers` | `designed / caption / seed` を追加（seed は文字列）。既存フィールドは変更なし |

## 🤖 原稿AI と 絵文字ガイド（いろとりスタジオ 0.2）

### 原稿AI（ヘッダの「🤖 原稿AI」）
ローカルの CLI をヘッドレスで叩いて、caption（声の説明）と読ませる原稿を作らせるチャット。
**既定は ChatGPT（Codex）の `gpt-5.6-terra` / effort `low`**。Claude にも切り替えられる。

| | 使うもの |
|---|---|
| ChatGPT | `/Applications/ChatGPT.app/Contents/Resources/codex exec --skip-git-repo-check -m <model> -c model_reasoning_effort=<effort> -o <tmp> -`（プロンプトは stdin、答えは -o のファイルから読む。stdout には思考ログが混ざるため） |
| Claude | `~/.local/bin/claude -p --model <model>`（effort は `MAX_THINKING_TOKENS` で近似: none=1024 / low=2048 / medium=8192 / high=16384 / xhigh・max=31999） |

- **effort の有効値は `none / low / medium / high / xhigh / max`**。`minimal` は 400 で弾かれる
  （`Unsupported value: 'minimal' is not supported with the 'gpt-5.6-terra-1p-codexswic-ev3' model`）
- Rust 側 `ask_ai` / `ai_available`（main.rs）。CLAUDE.md の方針どおり **タイムアウト300秒 → kill → 2回までリトライ**
- **コマンドは必ず絶対パスで解決する**。GUI から起動したアプリの PATH は `/usr/bin:/bin:/usr/sbin:/sbin` しか無い。
  さらに `widen_path()` で `/opt/homebrew/bin`・`~/.local/bin` 等を前に足してから起動する
  （codex は細い PATH でも動くことを実測で確認済み。claude は環境を削ると "Not logged in" になったので補強してある）
- 会話履歴は CLI がステートレスなので毎回プロンプトに詰め直す（直近8往復）。
  **いま登録されている声とその caption も一緒に渡す**ので「さっきの先輩メカで」が通じる
- システムプロンプト `AI_SYSTEM`（index.html）に、caption の書き方・絵文字45種・110文字チャンク・
  方言のアクセント制約まで書いてある。**出力は必ず ``` で囲ませる**規約にしてあり、
  ブロックごとに「📝原稿へ / 🎬演技メモへ / 🎨声デザインへ / 📋コピー」で流し込める

### 絵文字ガイド（ヘッダの「⋯」）
`irodori_tts/gradio_emoji_palette.py` のパレット全45種を、感情／話し方・速さ／息・のど／口・動作の音／
空間・エフェクト の5群に分けて例文つきで一覧表示する。行をクリックすると原稿欄のカーソル位置に挿し込む。
