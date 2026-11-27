# PROGRESS

## いま何をやっているか
**.venv を Python 3.10 → 3.12 に引っ越した（2026-11-27、3.10 は 2026-10-01 でサポート終了）**。未コミット（`.python-version` と CONTEXT.md / PROGRESS.md）。

- 旧 venv の `uv pip freeze`（155個）を新しい 3.12 venv に `--no-deps` でそのまま入れた。版は全部同じ（freeze の diff ゼロ、`uv pip check` も通る）
- `sentencepiece==0.1.99` だけ 3.12 用 wheel が無いのでソースビルド。CMake 4 で落ちるので `CMAKE_POLICY_VERSION_MINIMUM=3.5` を付けた（ビルド済み wheel は uv キャッシュに残っとる）
- venv は `uv venv --relocatable` で作ってから `.venv` に改名（スクリプトに絶対パスが入らへん形）。旧 venv は `.venv-py310.bak` に置いてある
- `.python-version` を 3.12 に変えた（3.10 のままやと `uv run` が venv と食い違う）
- 確認: `say.sh` で 48kHz・5.28秒の WAV、`start_server.sh` で :3952 の /version と /synthesis（22050Hz・6.84秒）が返る、WebUI :3950 / :3951 も HTTP 200。確認後サーバと WebUI は止めた（もともと止まってた）

前回（2026-10-24、server 0.4.2）: スタジオに「キー（半音）」スライダーと頭の雑音削り・`denoise` を足した。.app は差し替え済み。声DBに「ヘリウム男子」と「たかたか（超高音）」あり

## 次にやること
- [ ] しばらく 3.12 で普段どおり使って問題なければ `.venv-py310.bak` を消し、`uv python uninstall 3.10` する（3.10 を使ってるのは今この bak だけ）
- [ ] 3.12 引っ越し分（`.python-version`・CONTEXT.md・PROGRESS.md）をコミット
- [ ] たかたかがスタジオを開き直して「▶ サーバを起動」→ キーのスライダーで試し聞き。高さと頭の雑音を耳で確認 → OK ならコミット
- [ ] 「たかたか（超高音）」を消すか決める（たかたか判断）
- [ ] 原稿AIの既定 Codex が動かへん（ChatGPT.app 自体が未インストール）。入れ直すか、既定を Claude にするか、たかたか判断待ち
- [ ] econte 側で C1（256字）と C5（163字）を「録り直し」して、実際の音を耳で確認する
- [ ] 150ms のチャンク間無音・/render の 350ms・区切り110字を、長文を何本か聞いて調整する
- [ ] サーバを launchd 常駐にするか検討（`~/Library/LaunchAgents/com.stackchan.irodori.plist` はあるが今は読み込まれてへん）
- [ ] リポジトリ・ダッシュボードの 🔌ポートタブで 3950 / 3951 / 3952 を予約登録する
- [ ] ⋯メニューの seed 引き継ぎが「似た声のまま口調だけ変わる」か耳で確かめる
- [ ] 絵文字による感情制御を試して、どの絵文字がどう効くか手元メモを作る

## ハマっている点・未解決事項
- 3.12 に戻せへん事態になったら: `.venv` を別名にして `.venv-py310.bak` を `.venv` に戻し、`.python-version` を 3.10 に戻す（旧 venv は元の場所の名前に戻せば動く）
- 3.12 では descript-audiotools の docstring で `SyntaxWarning: invalid escape sequence` が出る。動作には関係なし
- torchcodec は Homebrew の ffmpeg 9（libavutil.60 より新しい）と合わず import できへん。3.10 の頃から同じで、推論・サーバ・WebUI は使ってへん
- 雑音は声の最中にも乗っとる（声に隠れて聞こえへんはず）。それも気になるなら手本（refs/takataka_*）の録り直しが根治
- `say.sh` には雑音消しを入れてへん（キーだけ）
- -12（下げる方向）は長さだけ確認。高さは測る道具が 60Hz 未満を拾えず未計測
- **30秒の壁はモデル側の仕様**。読点も句点も無い超長文は `split_script` がそのまま1チャンクで通すので無防備
- チャンクを跨ぐと声の調子が微妙に変わる可能性がある。繋ぎ目の自然さは耳で確認が要る
- econte 側のキャッシュは行の文章が変わらん限り効くので、サーバを直しても録り直さんと古い音のまま
