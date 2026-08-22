"""econte など外部アプリから叩くための最小HTTPサーバ。

Gradio を経由せずにモデルを常駐させ、参照音声の latent を起動時に一度だけ焼く。
これで1リクエストあたりのモデルロード（約6秒）と参照エンコード（約4秒）が消える。

エンドポイント（VOICEVOX ENGINE の作法に寄せてある。econte が同じ形で扱えるように）:
  GET  /version    -> {"version": "...", "device": "mps", "checkpoint": "..."}
  GET  /speakers   -> [{"id": "takataka", "name": "たかたか（本人）", "clips": 3}]
  POST /synthesis  -> リクエストJSONを受けて 22050Hz/モノラル/16bit の WAV バイト列を返す
                      {speaker, text, expression, rate, take?}  take>0 で別テイク（録り直し）

合成は 1 件ずつ直列に流す（MPS 上のモデルを複数スレッドで同時に叩かせない）。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import threading
import time
import wave
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torchaudio

from irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    _load_audio,
    default_runtime_device,
    download_hf_checkpoint,
    get_cached_runtime,
)

REPO_ROOT = Path(__file__).resolve().parent
SPEAKERS_JSON = REPO_ROOT / "speakers.json"
LATENT_DIR = REPO_ROOT / "refs" / ".latents"

# econte 側のナレーション連結パイプラインがこの形式しか受け取らない
OUT_SAMPLE_RATE = 22050

# 抑揚3段階。Irodori は数値パラメータではなく日本語のキャプションで演技を指示する。
# flat はキャプション無し＝素の読みにして、キャプション条件付けの揺れを持ち込まない。
EXPRESSION_CAPTIONS = {
    "flat": None,
    "natural": "自然な抑揚で、人が普通に話しているくらいの落ち着いた調子で読み上げてください。",
    "rich": "感情豊かに、抑揚を大きくつけて、聞き手に語りかけるように読み上げてください。",
}

DEFAULT_SPEAKERS = {
    "takataka": {
        "name": "たかたか（本人）",
        "clips": ["refs/takataka_a.wav", "refs/takataka_b.wav", "refs/takataka_c.wav"],
    }
}

log = logging.getLogger("irodori.server")


@dataclass
class Speaker:
    id: str
    name: str
    clips: list[Path]
    latents: list[Path]


def load_speaker_config() -> dict:
    """speakers.json があればそれを、無ければ既定（たかたか本人）を使う。"""
    if not SPEAKERS_JSON.exists():
        return DEFAULT_SPEAKERS
    try:
        raw = json.loads(SPEAKERS_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        # 壊れた設定を黙って既定に差し替えると「なぜ声が変わったか」が分からなくなる
        raise SystemExit(f"speakers.json が読めません: {e}") from e
    if not isinstance(raw, dict) or not raw:
        raise SystemExit("speakers.json が空、または辞書ではありません")
    return raw


def _clip_fingerprint(clips: list[Path]) -> str:
    """参照wavの中身が変わったら latent を焼き直すための指紋。パス・サイズ・更新時刻で見る。"""
    h = hashlib.sha256()
    for p in clips:
        st = p.stat()
        h.update(str(p).encode("utf-8"))
        h.update(str(st.st_size).encode("utf-8"))
        h.update(str(int(st.st_mtime)).encode("utf-8"))
    return h.hexdigest()[:16]


def _resolve_checkpoint_path(checkpoint: str) -> str:
    """ローカルの .pt/.safetensors はそのまま、HF のリポIDはダウンロードして実体パスに変える。"""
    if Path(checkpoint).suffix in {".pt", ".safetensors"}:
        return checkpoint
    resolved = str(download_hf_checkpoint(checkpoint))
    log.info("checkpoint: hf://%s -> %s", checkpoint, resolved)
    return resolved


class SynthServer:
    def __init__(self, checkpoint: str, device: str, num_steps: int, codec_repo: str | None):
        self.checkpoint = checkpoint
        self.device = device
        self.num_steps = num_steps
        key_args = {
            "checkpoint": _resolve_checkpoint_path(checkpoint),
            "model_device": device,
            "codec_device": device,
        }
        if codec_repo:
            key_args["codec_repo"] = codec_repo
        self.runtime, _ = get_cached_runtime(RuntimeKey(**key_args))
        self.lock = threading.Lock()
        self.speakers: dict[str, Speaker] = {}
        self._prepare_speakers()

    def _prepare_speakers(self) -> None:
        """参照wavを latent に焼いて保存する。既に同じ指紋の latent があれば再利用。"""
        LATENT_DIR.mkdir(parents=True, exist_ok=True)
        for spk_id, cfg in load_speaker_config().items():
            clips = [(REPO_ROOT / c).resolve() for c in cfg.get("clips", [])]
            missing = [str(c) for c in clips if not c.exists()]
            if not clips or missing:
                log.warning("話者 %s をとばします（参照音声が無い: %s）", spk_id, missing or "未指定")
                continue
            fp = _clip_fingerprint(clips)
            latents: list[Path] = []
            for i, clip in enumerate(clips):
                out = LATENT_DIR / f"{spk_id}_{fp}_{i}.pt"
                if not out.exists():
                    t0 = time.perf_counter()
                    wav, sr = _load_audio(clip)
                    piece = self.runtime.codec.encode_waveform(
                        wav.unsqueeze(0),
                        sample_rate=int(sr),
                        normalize_db=-16.0,
                        ensure_max=True,
                    ).cpu()
                    if piece.shape[1] == 0:
                        raise SystemExit(f"参照音声が空の latent になりました: {clip}")
                    # 読み出し側は (T, D) の2次元を想定している
                    torch.save(piece.squeeze(0).contiguous(), out)
                    log.info(
                        "latent を作成: %s (%.2fs, %d steps)",
                        out.name,
                        time.perf_counter() - t0,
                        piece.shape[1],
                    )
                latents.append(out)
            self.speakers[spk_id] = Speaker(
                id=spk_id,
                name=cfg.get("name", spk_id),
                clips=clips,
                latents=latents,
            )
        if not self.speakers:
            raise SystemExit(
                "使える話者がひとつもありません。refs/ に参照音声を置くか speakers.json を直してください"
            )
        log.info("話者: %s", ", ".join(f"{s.id}({len(s.clips)}本)" for s in self.speakers.values()))

    def synthesize(self, params: dict) -> bytes:
        text = str(params.get("text", "")).strip()
        if not text:
            raise ValueError("text が空です")
        spk_id = str(params.get("speaker") or next(iter(self.speakers)))
        speaker = self.speakers.get(spk_id)
        if speaker is None:
            raise ValueError(f"知らない話者です: {spk_id}（使えるのは {list(self.speakers)}）")

        expression = str(params.get("expression", "flat"))
        if expression not in EXPRESSION_CAPTIONS:
            raise ValueError(f"知らない抑揚です: {expression}")
        caption = EXPRESSION_CAPTIONS[expression]

        # 呼び出し側は say の rate(=180が等速) で速さを持っている。
        # duration_scale は「尺の倍率」なので、速くしたい = 尺を縮める = 180/rate。
        rate = float(params.get("rate", 180.0))
        duration_scale = 1.0 if rate <= 0 else max(0.6, min(1.8, 180.0 / rate))

        num_steps = int(params.get("num_steps", self.num_steps))

        # 同じ文章は毎回同じ音になってほしい（呼び出し側が行キャッシュを持つため、
        # キャッシュを消して作り直したときに音が変わるとつなぎ目で違和感が出る）。
        # ただし呼び出し側が「録り直し」を明示したとき（take>0）は別のテイクを返す。
        take = int(params.get("take", 0) or 0)
        seed_src = f"{spk_id}|{expression}|{rate}|{num_steps}|{text}|{take}"
        seed = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:15], 16)

        req = SamplingRequest(
            text=text,
            caption=caption,
            ref_latents=[str(p) for p in speaker.latents],
            duration_scale=duration_scale,
            num_steps=num_steps,
            seed=seed,
        )
        with self.lock:  # モデルは1本しかないので同時実行させない
            t0 = time.perf_counter()
            result = self.runtime.synthesize(req)
            elapsed = time.perf_counter() - t0

        wav_bytes = to_wav_bytes(result.audio, result.sample_rate)
        log.info(
            "合成 %.2fs speaker=%s expr=%s steps=%d chars=%d -> %.2fs音声",
            elapsed,
            spk_id,
            expression,
            num_steps,
            len(text),
            len(wav_bytes) / (OUT_SAMPLE_RATE * 2),
        )
        return wav_bytes


def to_wav_bytes(audio: torch.Tensor, sample_rate: int) -> bytes:
    """生成音を 22050Hz・モノラル・16bit の WAV バイト列にする。"""
    wav = audio.detach().to("cpu", dtype=torch.float32)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:  # 念のため（v4-Smallはモノラルを返す）
        wav = wav.mean(dim=0, keepdim=True)
    if sample_rate != OUT_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sample_rate, OUT_SAMPLE_RATE)
    # リサンプル後に 1.0 を超えると int16 で折り返してバリッと歪むので、超えた分だけ縮める
    peak = float(wav.abs().max())
    if peak > 1.0:
        wav = wav / peak
    pcm = (wav.clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16).numpy().tobytes()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(OUT_SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    server_version = "IrodoriTTS/0.1"
    synth: SynthServer  # ServerRunner が差し込む

    def log_message(self, fmt, *args):  # 既定の stderr 直書きを logging に寄せる
        log.debug("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, code: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler の規約)
        if self.path.split("?")[0] == "/version":
            self._send_json(
                200,
                {
                    "version": "0.1.0",
                    "engine": "irodori-tts",
                    "device": self.synth.device,
                    "checkpoint": self.synth.checkpoint,
                    "sample_rate": OUT_SAMPLE_RATE,
                },
            )
            return
        if self.path.split("?")[0] == "/speakers":
            self._send_json(
                200,
                [
                    {"id": s.id, "name": s.name, "clips": len(s.clips)}
                    for s in self.synth.speakers.values()
                ],
            )
            return
        self._send_json(404, {"error": f"知らないパスです: {self.path}"})

    def do_POST(self):  # noqa: N802
        if self.path.split("?")[0] != "/synthesis":
            self._send_json(404, {"error": f"知らないパスです: {self.path}"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send_json(400, {"error": "Content-Length が読めません"})
            return
        if length <= 0:
            self._send_json(400, {"error": "リクエストボディが空です"})
            return
        try:
            params = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            self._send_json(400, {"error": f"リクエストJSONが読めません: {e}"})
            return
        if not isinstance(params, dict):
            self._send_json(400, {"error": "リクエストJSONは辞書で送ってください"})
            return

        try:
            wav_bytes = self.synth.synthesize(params)
        except ValueError as e:
            self._send_json(400, {"error": str(e)})
            return
        except Exception as e:  # 落とさずに理由を返す。サーバが死ぬと呼び出し側が全部止まる
            log.exception("合成に失敗")
            self._send_json(500, {"error": f"合成に失敗しました: {e}"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav_bytes)))
        self.end_headers()
        self.wfile.write(wav_bytes)


def main() -> None:
    ap = argparse.ArgumentParser(description="Irodori-TTS の常駐HTTPサーバ")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3952)
    ap.add_argument("--checkpoint", default="Aratako/Irodori-TTS-v4.1-Small")
    ap.add_argument("--device", default=None, help="既定は自動検出（Macなら mps）")
    ap.add_argument("--codec-repo", default=None)
    ap.add_argument(
        "--num-steps",
        type=int,
        default=32,
        help="サンプリング回数。既定40より少し下げて仮ナレ向けに速度を取っている",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    device = args.device or default_runtime_device()
    log.info("モデルを読み込み中… checkpoint=%s device=%s", args.checkpoint, device)
    t0 = time.perf_counter()
    synth = SynthServer(args.checkpoint, device, args.num_steps, args.codec_repo)
    log.info("準備完了（%.1fs）", time.perf_counter() - t0)

    Handler.synth = synth
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info("待ち受け開始: http://%s:%d", args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("停止します")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
