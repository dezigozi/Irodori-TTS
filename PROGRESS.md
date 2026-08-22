# PROGRESS

## いま何をやっているか
Irodori-TTS を MacBook（M5 / 32GB / macOS 26.6.2）でローカル運用できる状態にした。完了済み。

- `~/git/Irodori-TTS` に clone、`uv sync --extra cpu` で環境構築（Python 3.10.20 / torch 2.10.0 / MPS 利用可）
- `Aratako/Irodori-TTS-v4.1-Small` をDLして CLI 推論を確認（VoiceDesign / 参照音声クローンの両方）
- たかたかの声から参照クリップ 10秒×3本を `refs/` に作成しクローン生成に成功。
  素材を `自分の音声.m4a`（ALAC 48kHz/24bit）に差し替え、生成音の8kHz以上が +7.5dB 改善（旧mp3版は `refs/old_mp3/` に退避）
- `start_webui.sh`（:3950）/ `start_webui_voicedesign.sh`（:3951）/ `say.sh` を追加、いずれも動作確認済み
- **常駐HTTPサーバ `server.py`（:3952）を追加** → econte（ぷれぜん君）の4本目のナレーションエンジンとして連携済み。
  モデル常駐＋latent事前計算で1行9秒前後。22050Hz/mono/16bitで返すのでeconte側は変換不要
- ポート 3950・3951・3952 を `~/.claude/PORTS.md` に登録済み

## 次にやること
- [ ] リポジトリ・ダッシュボードの 🔌ポートタブで 3950 / 3951 を予約登録する（PORTS.md は更新済み、ダッシュボード側が未登録）
- [ ] econteで実際に9カットぶんの仮ナレを作って、ElevenLabsと聞き比べ→乗り換えるか決める
- [ ] 絵文字による感情制御を実際に試してどの絵文字がどう効くか手元メモを作る
- [ ] サーバをlaunchd常駐にするか検討（今はeconteの「確認・起動」ボタンで都度立ち上げ）
- [ ] よく使うならたかたかポータルに登録（WebUI 起動の .command ラッパーを作る）

## ハマっている点・未解決事項
- 特になし（参照音声のリテイクは完了）
- MPS では fp32 のみ。bf16 高速化は CUDA/XPU 専用なので Mac では効かない
