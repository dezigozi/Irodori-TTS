"""econte や「いろとりスタジオ」(app/) など外部アプリから叩くための常駐HTTPサーバ。

Gradio を経由せずにモデルを常駐させ、参照音声の latent を起動時に一度だけ焼く。
これで1リクエストあたりのモデルロード（約6秒）と参照エンコード（約4秒）が消える。

エンドポイント（VOICEVOX ENGINE の作法に寄せてある。econte が同じ形で扱えるように）:
  GET  /version          -> {"version": "...", "device": "mps", "checkpoint": "...", "sample_rate": 22050}
  GET  /speakers         -> [{"id": "takataka", "name": "たかたか（本人）", "clips": 3, ...}]
  POST /speakers         -> 声の元ファイルから新しい話者を作る {name, source_path, clip_seconds?, max_clips?}
  PATCH  /speakers/<id>  -> 話者の表示名だけ変える {name}（音声・latent は焼き直さない）
  DELETE /speakers/<id>  -> 話者を消す（参照wav・latent も消す。takataka は保護）
  POST /synthesis        -> 22050Hz/モノラル/16bit の WAV バイト列を返す（econte 互換）
                            {speaker, text, expression, rate, take?, caption?, sample_rate?}
  POST /render           -> 長文を分割→合成→連結して WAV ファイルに保存するジョブを開始
                            {speaker, text, expression|caption, rate, num_steps?, out_path, pause_ms?}
  GET  /jobs/<id>        -> ジョブ進捗 {status, done, total, out_path, seconds, error}
  POST /jobs/<id>/cancel -> ジョブ中断（チャンク境界で止まる）

合成は 1 件ずつ直列に流す（MPS 上のモデルを複数スレッドで同時に叩かせない）。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

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
REFS_DIR = REPO_ROOT / "refs"
LATENT_DIR = REFS_DIR / ".latents"

# econte 側のナレーション連結パイプラインがこの形式しか受け取らない
OUT_SAMPLE_RATE = 22050
# いろとりスタジオの保存用（モデル素の 48kHz）
NATIVE_SAMPLE_RATE = 48000
ALLOWED_SAMPLE_RATES = {OUT_SAMPLE_RATE, NATIVE_SAMPLE_RATE}

# 消してはいけない話者（本人の声）
PROTECTED_SPEAKERS = {"takataka"}

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

# 声デザイン（VoiceDesign）: 参照音声を使わず caption だけで声を作るときに読ませる文。
# 試し聞きは短く、参照クリップ用は「10秒 × 2〜3本」に切れる長さで、いろんな音素が出るようにする。
DESIGN_PREVIEW_TEXT = "こんにちは。この声で原稿を読み上げます。よろしくお願いします。"
DESIGN_REF_TEXT = (
    "こんにちは。今日はいい天気ですね。わたしはこの声で、原稿を読み上げます。"
    "数字は、一、二、三、四、五。ゆっくり、はっきりと、落ち着いて話しています。"
    "どうぞ、よろしくお願いします。"
)
# モデルの上限が30秒。参照用は長めに出したいので目一杯まで許す
DESIGN_MAX_SECONDS = 30.0

# 1チャンクの上限文字数（max_text_len=256トークン / max_seconds=30 の安全圏）。
# モデルは尺を max_seconds=30 でクランプする（inference_runtime の latent_steps）ので、
# 30秒を超える文章を丸ごと投げると**枠に押し込むぶん早口になる**。110字はその安全圏。
# 長文レンダー（/render）だけやなく、1発合成（/synthesis）もここで切る
RENDER_CHUNK_CHARS = 110
# /synthesis で切ったときの、チャンク間に挟む無音（ms）。
# 句読点で切っているので、レンダー（350ms＝段落の間）よりは詰めて自然に繋ぐ
SYNTH_CHUNK_PAUSE_MS = 150

log = logging.getLogger("irodori.server")


@dataclass
class Speaker:
    id: str
    name: str
    clips: list[Path]
    latents: list[Path]
    source: str | None = None
    created_at: int | None = None
    # 声デザインで作った話者だけ入る。同じ声をもう一度作り直すための種
    caption: str | None = None
    seed: int | None = None

    def to_public(self) -> dict:
        # 先頭3つは econte が読む形。あとはスタジオ向けの追加情報（足すだけで形は崩さない）
        return {
            "id": self.id,
            "name": self.name,
            "clips": len(self.clips),
            "clip_paths": [str(p) for p in self.clips],
            "source": self.source,
            "created_at": self.created_at,
            "protected": self.id in PROTECTED_SPEAKERS,
            "designed": self.caption is not None,
            "caption": self.caption,
            # JS の Number では 2^53 を超えると精度が落ちるので必ず文字列で返す
            "seed": None if self.seed is None else str(self.seed),
        }


@dataclass
class RenderJob:
    id: str
    out_path: Path
    total: int = 0
    done: int = 0
    status: str = "queued"  # queued | running | done | error | cancelled
    error: str = ""
    seconds: float = 0.0
    cancel: threading.Event = field(default_factory=threading.Event)
    # 声デザイン登録ジョブが、できあがった話者を返すのに使う
    result: dict | None = None
    label: str = "render"

    def to_public(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "done": self.done,
            "total": self.total,
            "out_path": str(self.out_path),
            "seconds": round(self.seconds, 2),
            "error": self.error,
            "label": self.label,
            "result": self.result,
        }


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


def save_speaker_config(cfg: dict) -> None:
    """speakers.json を原子的に書き換える（途中で落ちても半端なJSONを残さない）。"""
    tmp = SPEAKERS_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, SPEAKERS_JSON)


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


# ---------------------------------------------------------------------------
# 声の元ファイル → 参照クリップ（ffmpeg）
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: float, label: str) -> subprocess.CompletedProcess:
    """外部CLIは必ずタイムアウト付きで呼ぶ（終わらないプロセスにサーバごと道連れにされないため）。"""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"{label} がタイムアウトしました（{timeout:.0f}秒）") from e
    except FileNotFoundError as e:
        raise RuntimeError(f"{label} が見つかりません（{cmd[0]}）。brew install ffmpeg してください") from e


def probe_audio(path: Path) -> tuple[float, int]:
    """ffprobe で (尺[秒], サンプルレート) を返す。"""
    cp = _run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate:format=duration",
            "-of", "json", str(path),
        ],
        timeout=30,
        label="ffprobe",
    )
    if cp.returncode != 0:
        raise ValueError(f"音声として読めませんでした: {cp.stderr.strip()[:200]}")
    info = json.loads(cp.stdout or "{}")
    streams = info.get("streams") or []
    if not streams:
        raise ValueError("音声ストリームが入っていません")
    sr = int(streams[0].get("sample_rate") or 0)
    duration = float((info.get("format") or {}).get("duration") or 0.0)
    if sr <= 0 or duration <= 0:
        raise ValueError("尺かサンプルレートが取れませんでした")
    return duration, sr


def _mean_volume_db(path: Path) -> float:
    cp = _run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        timeout=60,
        label="ffmpeg volumedetect",
    )
    m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", cp.stderr)
    return float(m.group(1)) if m else -99.0


def build_reference_clips(
    spk_id: str, source: Path, clip_seconds: float = 10.0, max_clips: int = 3
) -> tuple[list[Path], float, int]:
    """声の元ファイルから refs/<id>_full.wav と refs/<id>_a.wav… を作る。

    CONTEXT.md の知見どおり「10秒 × 最大3本」。元のサンプルレートは維持する
    （コーデックが内部で1回だけリサンプルするので、こちらで落とすと二重変換になる）。
    48kHz 超だけは 48kHz に落とす。
    """
    duration, src_sr = probe_audio(source)
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    out_sr = min(src_sr, NATIVE_SAMPLE_RATE)

    full = REFS_DIR / f"{spk_id}_full.wav"
    cp = _run(
        ["ffmpeg", "-hide_banner", "-y", "-i", str(source), "-vn", "-ac", "1",
         "-ar", str(out_sr), "-c:a", "pcm_s24le", str(full)],
        timeout=600,
        label="ffmpeg 変換",
    )
    if cp.returncode != 0 or not full.exists():
        raise ValueError(f"wav への変換に失敗しました: {cp.stderr.strip()[-300:]}")

    clips: list[Path] = []
    if duration <= clip_seconds + 1.0:
        # 短い素材は丸ごと1本
        clip = REFS_DIR / f"{spk_id}_a.wav"
        shutil.copyfile(full, clip)
        clips.append(clip)
        return clips, duration, out_sr

    n = max(1, min(max_clips, int(duration // clip_seconds)))
    # 先頭1秒と末尾は避けて均等配置（先頭はノイズ・無音が多い）
    usable = duration - clip_seconds - 1.0
    for i in range(n):
        offset = 1.0 + (usable * i / max(1, n - 1) if n > 1 else usable / 2)
        clip = REFS_DIR / f"{spk_id}_{chr(ord('a') + i)}.wav"
        for attempt in range(3):
            cp = _run(
                ["ffmpeg", "-hide_banner", "-y", "-ss", f"{offset:.2f}", "-t", f"{clip_seconds:.2f}",
                 "-i", str(full), "-c:a", "pcm_s24le", str(clip)],
                timeout=120,
                label="ffmpeg 切り出し",
            )
            if cp.returncode != 0 or not clip.exists():
                raise ValueError(f"クリップの切り出しに失敗しました: {cp.stderr.strip()[-300:]}")
            vol = _mean_volume_db(clip)
            if vol >= -45.0:
                break
            # ほぼ無音のクリップは5秒ずらして取り直す
            log.info("クリップ %s は無音気味（%.1f dB）。5秒ずらします", clip.name, vol)
            offset = min(offset + 5.0, duration - clip_seconds)
        clips.append(clip)
    return clips, duration, out_sr


# ---------------------------------------------------------------------------
# 長文の分割
# ---------------------------------------------------------------------------

def split_script(text: str, max_chars: int = RENDER_CHUNK_CHARS) -> list[str]:
    """原稿を文単位で切って、max_chars を超えない範囲でまとめる。空行は段落の切れ目として尊重。"""
    chunks: list[str] = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = para.strip()
        if not para:
            continue
        sentences = [s.strip() for s in re.split(r"(?<=[。！？!?\n])", para) if s.strip()]
        buf = ""
        for s in sentences:
            # 1文が長すぎるときは読点でさらに割る
            parts = [s] if len(s) <= max_chars else [
                p for p in re.split(r"(?<=[、,])", s) if p.strip()
            ]
            for p in parts:
                if buf and len(buf) + len(p) > max_chars:
                    chunks.append(buf)
                    buf = ""
                buf += p
                if len(buf) > max_chars:  # 読点もない超長文はそのまま1チャンク
                    chunks.append(buf)
                    buf = ""
        if buf:
            chunks.append(buf)
    return chunks


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
        self.lock = threading.Lock()  # モデル（MPS）を触るときは必ずこれを取る
        self.cfg_lock = threading.Lock()  # speakers.json と self.speakers の書き換え
        self.speakers: dict[str, Speaker] = {}
        self.jobs: dict[str, RenderJob] = {}
        self.jobs_lock = threading.Lock()
        self._prepare_speakers()

    # ---------------- 話者 ----------------

    def _prepare_one(self, spk_id: str, cfg: dict) -> Speaker | None:
        """1話者ぶんの参照wavを latent に焼く。既に同じ指紋の latent があれば再利用。"""
        clips = [(REPO_ROOT / c).resolve() for c in cfg.get("clips", [])]
        missing = [str(c) for c in clips if not c.exists()]
        if not clips or missing:
            log.warning("話者 %s をとばします（参照音声が無い: %s）", spk_id, missing or "未指定")
            return None
        fp = _clip_fingerprint(clips)
        latents: list[Path] = []
        for i, clip in enumerate(clips):
            out = LATENT_DIR / f"{spk_id}_{fp}_{i}.pt"
            if not out.exists():
                t0 = time.perf_counter()
                wav, sr = _load_audio(clip)
                with self.lock:
                    piece = self.runtime.codec.encode_waveform(
                        wav.unsqueeze(0),
                        sample_rate=int(sr),
                        normalize_db=-16.0,
                        ensure_max=True,
                    ).cpu()
                if piece.shape[1] == 0:
                    raise ValueError(f"参照音声が空の latent になりました: {clip}")
                # 読み出し側は (T, D) の2次元を想定している
                torch.save(piece.squeeze(0).contiguous(), out)
                log.info(
                    "latent を作成: %s (%.2fs, %d steps)",
                    out.name,
                    time.perf_counter() - t0,
                    piece.shape[1],
                )
            latents.append(out)
        return Speaker(
            id=spk_id,
            name=cfg.get("name", spk_id),
            clips=clips,
            latents=latents,
            source=cfg.get("source"),
            created_at=cfg.get("created_at"),
            caption=cfg.get("caption"),
            seed=cfg.get("seed"),
        )

    def _prepare_speakers(self) -> None:
        LATENT_DIR.mkdir(parents=True, exist_ok=True)
        for spk_id, cfg in load_speaker_config().items():
            try:
                spk = self._prepare_one(spk_id, cfg)
            except ValueError as e:
                raise SystemExit(str(e)) from e
            if spk:
                self.speakers[spk_id] = spk
        if not self.speakers:
            # 昔はここで落としていたが、声デザイン（参照音声なし）で話者を作れるようになったので
            # 起動は通す。clone 直後や新しいマシンでも、まず声をデザインしてもらえばいい。
            log.warning(
                "話者がひとつもありません。いろとりスタジオの「🎨 声をデザインして作る」か、"
                "声の元ファイルのドロップで作ってください（POST /design/speakers・POST /speakers）"
            )
            return
        log.info("話者: %s", ", ".join(f"{s.id}({len(s.clips)}本)" for s in self.speakers.values()))

    def add_speaker(self, params: dict) -> dict:
        name = str(params.get("name", "")).strip()
        src_raw = str(params.get("source_path", "")).strip()
        if not name:
            raise ValueError("name（表示名）が空です")
        if not src_raw:
            raise ValueError("source_path（声の元ファイル）が空です")
        source = Path(os.path.expanduser(src_raw)).resolve()
        if not source.is_file():
            raise ValueError(f"声の元ファイルが見つかりません: {source}")
        clip_seconds = float(params.get("clip_seconds", 10.0) or 10.0)
        max_clips = int(params.get("max_clips", 3) or 3)
        if not (3.0 <= clip_seconds <= 30.0):
            raise ValueError("clip_seconds は 3〜30 秒にしてください")
        if not (1 <= max_clips <= 6):
            raise ValueError("max_clips は 1〜6 にしてください")

        spk_id = f"spk_{int(time.time())}"
        with self.cfg_lock:
            cfg_all = dict(load_speaker_config())
            while spk_id in cfg_all or spk_id in self.speakers:
                spk_id = f"spk_{int(time.time() * 1000)}"
            t0 = time.perf_counter()
            clips, duration, sr = build_reference_clips(spk_id, source, clip_seconds, max_clips)
            log.info("参照クリップ %d本を作成（元 %.1f秒 / %dHz, %.1fs）", len(clips), duration, sr,
                     time.perf_counter() - t0)
            cfg = {
                "name": name,
                "clips": [str(c.relative_to(REPO_ROOT)) for c in clips],
                "source": str(source),
                "created_at": int(time.time()),
            }
            try:
                spk = self._prepare_one(spk_id, cfg)
            except Exception:
                # 焼けなかった話者のゴミを残さない
                self._remove_speaker_files(spk_id)
                raise
            if spk is None:
                self._remove_speaker_files(spk_id)
                raise ValueError("参照クリップが作れませんでした")
            cfg_all[spk_id] = cfg
            save_speaker_config(cfg_all)
            self.speakers[spk_id] = spk
        pub = spk.to_public()
        pub.update({"duration": round(duration, 1), "sample_rate": sr})
        log.info("話者を追加: %s (%s)", spk_id, name)
        return pub

    def _remove_speaker_files(self, spk_id: str) -> None:
        """refs/<id>_*.wav と latent を消す。refs 配下の絶対パスだけを対象にする。"""
        for p in list(REFS_DIR.glob(f"{spk_id}_*.wav")) + list(LATENT_DIR.glob(f"{spk_id}_*.pt")):
            p = p.resolve()
            if REFS_DIR.resolve() in p.parents and p.is_file():
                p.unlink()
                log.info("削除: %s", p)

    def rename_speaker(self, spk_id: str, new_name: str) -> dict:
        # 表示名を差し替えるだけ。参照クリップも latent も触らんので焼き直しは要らん。
        name = str(new_name or "").strip()
        if not name:
            raise ValueError("name（表示名）が空です")
        if len(name) > 40:
            raise ValueError("name（表示名）は40文字までにしてください")
        with self.cfg_lock:
            cfg_all = dict(load_speaker_config())
            if spk_id not in cfg_all:
                raise ValueError(f"知らない話者です: {spk_id}")
            old = str(cfg_all[spk_id].get("name") or spk_id)
            cfg_all[spk_id] = {**cfg_all[spk_id], "name": name}
            save_speaker_config(cfg_all)
            spk = self.speakers.get(spk_id)
            if spk is not None:
                spk.name = name
        log.info("話者の名前を変更: %s（%s → %s）", spk_id, old, name)
        spk = self.speakers.get(spk_id)
        return spk.to_public() if spk is not None else {"id": spk_id, "name": name}

    def delete_speaker(self, spk_id: str) -> dict:
        if spk_id in PROTECTED_SPEAKERS:
            raise ValueError(f"{spk_id} は保護されている話者なので消せません")
        with self.cfg_lock:
            cfg_all = dict(load_speaker_config())
            if spk_id not in cfg_all and spk_id not in self.speakers:
                raise ValueError(f"知らない話者です: {spk_id}")
            cfg_all.pop(spk_id, None)
            save_speaker_config(cfg_all)
            self.speakers.pop(spk_id, None)
            self._remove_speaker_files(spk_id)
        log.info("話者を削除: %s", spk_id)
        return {"deleted": spk_id}

    # ---------------- 合成 ----------------

    def _build_request(self, params: dict, text: str) -> tuple[SamplingRequest, Speaker]:
        if not self.speakers:
            raise ValueError(
                "話者がひとつもありません。先に「🎨 声をデザインして作る」か"
                "声の元ファイルのドロップで声を作ってください"
            )
        spk_id = str(params.get("speaker") or next(iter(self.speakers)))
        speaker = self.speakers.get(spk_id)
        if speaker is None:
            raise ValueError(f"知らない話者です: {spk_id}（使えるのは {list(self.speakers)}）")

        # caption（自由文）が来ていればそれを優先。無ければ expression 3段階。
        caption = str(params.get("caption") or "").strip() or None
        expression = str(params.get("expression", "flat"))
        if caption is None:
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
        seed_src = f"{spk_id}|{expression}|{caption or ''}|{rate}|{num_steps}|{text}|{take}"
        seed = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:15], 16)

        req = SamplingRequest(
            text=text,
            caption=caption,
            ref_latents=[str(p) for p in speaker.latents],
            duration_scale=duration_scale,
            num_steps=num_steps,
            seed=seed,
        )
        return req, speaker

    # ---------------- 声デザイン（参照音声なし） ----------------
    # 実在の人の声を1本も使わずに、caption（どんな声かの説明文）だけで声を作る。
    # 同じ caption でも seed が変われば別人になるので、試し聞き→気に入った seed を登録、という流れ。

    def _build_design_request(self, params: dict, text: str) -> tuple[SamplingRequest, int]:
        caption = str(params.get("caption") or "").strip()
        if not caption:
            raise ValueError("caption（どんな声にしたいかの説明）が空です")
        if not self.runtime.model_cfg.use_caption_condition:
            raise ValueError(
                "このチェックポイントは caption 条件づけに対応していないので声デザインは使えません"
            )

        rate = float(params.get("rate", 180.0))
        duration_scale = 1.0 if rate <= 0 else max(0.6, min(1.8, 180.0 / rate))
        num_steps = int(params.get("num_steps", self.num_steps))

        raw_seed = params.get("seed")
        if raw_seed is None or raw_seed == "":
            # 指定が無ければ毎回ちがう声（＝ガチャ）。使った種は必ず呼び出し側に返す
            seed = int.from_bytes(os.urandom(8), "big") >> 1
        else:
            try:
                seed = int(raw_seed)
            except (TypeError, ValueError) as e:
                raise ValueError(f"seed が数字ではありません: {raw_seed!r}") from e

        req = SamplingRequest(
            text=text,
            caption=caption,
            no_ref=True,
            duration_scale=duration_scale,
            num_steps=num_steps,
            max_seconds=DESIGN_MAX_SECONDS,
            # 参照音声が無いので話者ガイダンスは効かせない（本家の VoiceDesign UI と同じ扱い）
            cfg_scale_speaker=0.0,
            seed=seed,
        )
        return req, seed

    def _synth_design(self, params: dict, text: str) -> tuple[torch.Tensor, int, int]:
        req, seed = self._build_design_request(params, text)
        with self.lock:  # モデルは1本しかないので同時実行させない
            t0 = time.perf_counter()
            result = self.runtime.synthesize(req)
            elapsed = time.perf_counter() - t0
        log.info("声デザイン合成 %.2fs seed=%d steps=%d chars=%d",
                 elapsed, seed, req.num_steps, len(text))
        return result.audio, int(result.sample_rate), seed

    def design_preview(self, params: dict) -> tuple[bytes, int]:
        """試し聞き。使った seed を返すので、気に入ったらそれを登録に渡す。"""
        text = str(params.get("text") or "").strip() or DESIGN_PREVIEW_TEXT
        out_sr = int(params.get("sample_rate", NATIVE_SAMPLE_RATE) or NATIVE_SAMPLE_RATE)
        if out_sr not in ALLOWED_SAMPLE_RATES:
            raise ValueError(f"sample_rate は {sorted(ALLOWED_SAMPLE_RATES)} のどれかにしてください")
        audio, sr, seed = self._synth_design(params, text)
        return to_wav_bytes(audio, sr, out_sr), seed

    def start_design_speaker(self, params: dict) -> RenderJob:
        """デザインした声を、参照クリップを持つ普通の話者として登録するジョブを始める。

        参照クリップは「1回の生成を切り分けて」作る。文ごとに作り直すと声が揺れるので、
        1本の長い音声から切り出して声の同一性を担保する。
        """
        name = str(params.get("name", "")).strip()
        if not name:
            raise ValueError("name（表示名）が空です")
        if params.get("seed") is None or params.get("seed") == "":
            raise ValueError("seed が空です。先に試し聞きして、気に入った声の seed を渡してください")
        # caption・seed の妥当性はジョブを始める前に見る（開始後に落ちると気づきにくい）
        self._build_design_request(params, "テスト")

        with self.cfg_lock:
            cfg_all = load_speaker_config()
            spk_id = f"vox_{int(time.time())}"
            while spk_id in cfg_all or spk_id in self.speakers:
                spk_id = f"vox_{int(time.time() * 1000)}"

        job = RenderJob(
            id=uuid.uuid4().hex[:12],
            out_path=REFS_DIR / f"{spk_id}_full.wav",
            total=2,
            label="design",
        )
        with self.jobs_lock:
            self.jobs[job.id] = job
            for old in [j for j in self.jobs.values() if j.status in {"done", "error", "cancelled"}][:-50]:
                self.jobs.pop(old.id, None)
        threading.Thread(
            target=self._run_design_speaker, args=(job, spk_id, name, dict(params)),
            name=f"design-{job.id}", daemon=True,
        ).start()
        log.info("声デザイン登録を開始 job=%s id=%s name=%s", job.id, spk_id, name)
        return job

    def _run_design_speaker(self, job: RenderJob, spk_id: str, name: str, params: dict) -> None:
        job.status = "running"
        tmpdir: Path | None = None
        try:
            ref_text = str(params.get("ref_text") or "").strip() or DESIGN_REF_TEXT
            audio, sr, seed = self._synth_design(params, ref_text)
            job.done = 1
            if job.cancel.is_set():
                job.status = "cancelled"
                return

            wav = audio.detach().to("cpu", dtype=torch.float32)
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != NATIVE_SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, sr, NATIVE_SAMPLE_RATE)
            seconds = wav.shape[1] / NATIVE_SAMPLE_RATE
            if seconds < 3.0:
                raise ValueError(
                    f"参照にするには短すぎる音声（{seconds:.1f}秒）しか出ませんでした。"
                    "caption を変えるか、もう一度試してください"
                )

            # 切り出しは既存の build_reference_clips に任せる（元ファイルは一時領域に置く）
            tmpdir = Path(tempfile.mkdtemp(prefix="irodori-design-"))
            src = tmpdir / f"{spk_id}_src.wav"
            write_wav_file(src, wav, NATIVE_SAMPLE_RATE)

            clip_seconds = float(params.get("clip_seconds", 10.0) or 10.0)
            clip_seconds = max(3.0, min(clip_seconds, 30.0))
            max_clips = int(params.get("max_clips", 3) or 3)
            max_clips = max(1, min(max_clips, 6))

            with self.cfg_lock:
                cfg_all = dict(load_speaker_config())
                clips, duration, out_sr = build_reference_clips(spk_id, src, clip_seconds, max_clips)
                cfg = {
                    "name": name,
                    "clips": [str(c.relative_to(REPO_ROOT)) for c in clips],
                    "created_at": int(time.time()),
                    "caption": str(params.get("caption") or "").strip(),
                    "seed": seed,
                }
                try:
                    spk = self._prepare_one(spk_id, cfg)
                except Exception:
                    self._remove_speaker_files(spk_id)  # 焼けなかった話者のゴミを残さない
                    raise
                if spk is None:
                    self._remove_speaker_files(spk_id)
                    raise ValueError("参照クリップが作れませんでした")
                cfg_all[spk_id] = cfg
                save_speaker_config(cfg_all)
                self.speakers[spk_id] = spk

            job.done = 2
            job.seconds = duration
            job.result = spk.to_public()
            job.status = "done"
            log.info("声デザイン登録が完了 job=%s id=%s (%d本 / %.1f秒 / %dHz)",
                     job.id, spk_id, len(clips), duration, out_sr)
        except Exception as e:  # 理由をジョブに残す（握りつぶさない）
            log.exception("声デザイン登録に失敗 job=%s", job.id)
            self._remove_speaker_files(spk_id)
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
        finally:
            if tmpdir is not None:
                shutil.rmtree(tmpdir, ignore_errors=True)

    def _synth_tensor(self, params: dict, text: str) -> tuple[torch.Tensor, int, Speaker]:
        req, speaker = self._build_request(params, text)
        with self.lock:  # モデルは1本しかないので同時実行させない
            t0 = time.perf_counter()
            result = self.runtime.synthesize(req)
            elapsed = time.perf_counter() - t0
        log.info(
            "合成 %.2fs speaker=%s steps=%d chars=%d",
            elapsed, speaker.id, req.num_steps, len(text),
        )
        return result.audio, int(result.sample_rate), speaker

    @staticmethod
    def _to_native(audio: torch.Tensor, sr: int) -> torch.Tensor:
        """モノラル・NATIVE_SAMPLE_RATE に揃える（チャンクを繋ぐ前処理）"""
        wav = audio.detach().to("cpu", dtype=torch.float32)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != NATIVE_SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, NATIVE_SAMPLE_RATE)
        return wav

    def _synth_chunked(self, params: dict, text: str, pause_ms: int) -> tuple[torch.Tensor, int]:
        """長い文章は句読点で切って合成し、繋いで返す。
        切らずに投げると尺が max_seconds=30 でクランプされて早口になる
        （256字を1発で投げると約2倍速になった実測あり）。"""
        chunks = split_script(text)
        if len(chunks) <= 1:
            audio, sr, _ = self._synth_tensor(params, text)
            return audio, sr
        log.info("長文を分割して合成 chars=%d chunks=%d", len(text), len(chunks))
        pieces: list[torch.Tensor] = []
        for i, chunk in enumerate(chunks):
            audio, sr, _ = self._synth_tensor(params, chunk)
            pieces.append(self._to_native(audio, sr))
            if pause_ms > 0 and i < len(chunks) - 1:
                pieces.append(torch.zeros(1, int(NATIVE_SAMPLE_RATE * pause_ms / 1000)))
        return torch.cat(pieces, dim=1), NATIVE_SAMPLE_RATE

    def synthesize(self, params: dict) -> bytes:
        text = str(params.get("text", "")).strip()
        if not text:
            raise ValueError("text が空です")
        out_sr = int(params.get("sample_rate", OUT_SAMPLE_RATE) or OUT_SAMPLE_RATE)
        if out_sr not in ALLOWED_SAMPLE_RATES:
            raise ValueError(f"sample_rate は {sorted(ALLOWED_SAMPLE_RATES)} のどれかにしてください")
        pause_ms = int(params.get("pause_ms", SYNTH_CHUNK_PAUSE_MS) or 0)
        pause_ms = max(0, min(3000, pause_ms))
        audio, sr = self._synth_chunked(params, text, pause_ms)
        return to_wav_bytes(audio, sr, out_sr)

    # ---------------- 長文レンダー（ジョブ） ----------------

    def start_render(self, params: dict) -> RenderJob:
        text = str(params.get("text", "")).strip()
        if not text:
            raise ValueError("text が空です")
        out_raw = str(params.get("out_path", "")).strip()
        if not out_raw:
            raise ValueError("out_path（保存先）が空です")
        out_path = Path(os.path.expanduser(out_raw)).resolve()
        if out_path.suffix.lower() != ".wav":
            out_path = out_path.with_suffix(".wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # 話者・抑揚の妥当性はここで先に確かめる（ジョブ開始後に失敗すると気づきにくい）
        self._build_request(params, "テスト")
        chunks = split_script(text)
        if not chunks:
            raise ValueError("読み上げる文が見つかりません")
        pause_ms = int(params.get("pause_ms", 350) or 0)
        pause_ms = max(0, min(3000, pause_ms))

        job = RenderJob(id=uuid.uuid4().hex[:12], out_path=out_path, total=len(chunks))
        with self.jobs_lock:
            self.jobs[job.id] = job
            # 終わったジョブは50件まで残す
            for old in [j for j in self.jobs.values() if j.status in {"done", "error", "cancelled"}][:-50]:
                self.jobs.pop(old.id, None)
        threading.Thread(
            target=self._run_render, args=(job, chunks, dict(params), pause_ms),
            name=f"render-{job.id}", daemon=True,
        ).start()
        log.info("レンダー開始 job=%s chunks=%d -> %s", job.id, len(chunks), out_path)
        return job

    def _run_render(self, job: RenderJob, chunks: list[str], params: dict, pause_ms: int) -> None:
        job.status = "running"
        pieces: list[torch.Tensor] = []
        try:
            for i, chunk in enumerate(chunks):
                if job.cancel.is_set():
                    job.status = "cancelled"
                    log.info("レンダー中断 job=%s (%d/%d)", job.id, job.done, job.total)
                    return
                audio, sr, _ = self._synth_tensor(params, chunk)
                pieces.append(self._to_native(audio, sr))
                if pause_ms > 0 and i < len(chunks) - 1:
                    pieces.append(torch.zeros(1, int(NATIVE_SAMPLE_RATE * pause_ms / 1000)))
                job.done = i + 1
            full = torch.cat(pieces, dim=1)
            write_wav_file(job.out_path, full, NATIVE_SAMPLE_RATE)
            job.seconds = full.shape[1] / NATIVE_SAMPLE_RATE
            job.status = "done"
            log.info("レンダー完了 job=%s %.1f秒 -> %s", job.id, job.seconds, job.out_path)
        except Exception as e:  # 理由をジョブに残す（握りつぶさない）
            log.exception("レンダー失敗 job=%s", job.id)
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"

    def get_job(self, job_id: str) -> RenderJob:
        with self.jobs_lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise ValueError(f"知らないジョブです: {job_id}")
        return job


def _to_pcm16(audio: torch.Tensor, sample_rate: int, out_sr: int) -> bytes:
    wav = audio.detach().to("cpu", dtype=torch.float32)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:  # 念のため（v4-Smallはモノラルを返す）
        wav = wav.mean(dim=0, keepdim=True)
    if sample_rate != out_sr:
        wav = torchaudio.functional.resample(wav, sample_rate, out_sr)
    # リサンプル後に 1.0 を超えると int16 で折り返してバリッと歪むので、超えた分だけ縮める
    peak = float(wav.abs().max())
    if peak > 1.0:
        wav = wav / peak
    return (wav.clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16).numpy().tobytes()


def to_wav_bytes(audio: torch.Tensor, sample_rate: int, out_sr: int = OUT_SAMPLE_RATE) -> bytes:
    """生成音を out_sr・モノラル・16bit の WAV バイト列にする。"""
    pcm = _to_pcm16(audio, sample_rate, out_sr)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(out_sr)
        w.writeframes(pcm)
    return buf.getvalue()


def write_wav_file(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    """一時ファイルに書いてから置き換える（再生側が書きかけを掴まないように）。"""
    pcm = _to_pcm16(audio, sample_rate, sample_rate)
    fd, tmp = tempfile.mkstemp(prefix=".irodori_", suffix=".wav", dir=str(path.parent))
    os.close(fd)
    with wave.open(tmp, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    os.replace(tmp, path)


class Handler(BaseHTTPRequestHandler):
    server_version = "IrodoriTTS/0.3"
    synth: SynthServer  # ServerRunner が差し込む

    def log_message(self, fmt, *args):  # 既定の stderr 直書きを logging に寄せる
        log.debug("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, code: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send_json(400, {"error": "Content-Length が読めません"})
            return None
        if length <= 0:
            self._send_json(400, {"error": "リクエストボディが空です"})
            return None
        try:
            params = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            self._send_json(400, {"error": f"リクエストJSONが読めません: {e}"})
            return None
        if not isinstance(params, dict):
            self._send_json(400, {"error": "リクエストJSONは辞書で送ってください"})
            return None
        return params

    def _path(self) -> str:
        return unquote(urlparse(self.path).path)

    def do_OPTIONS(self):  # noqa: N802  Tauri の webview から fetch するための CORS 前飛行
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler の規約)
        path = self._path()
        if path == "/version":
            self._send_json(
                200,
                {
                    "version": "0.3.2",
                    "engine": "irodori-tts",
                    "device": self.synth.device,
                    "checkpoint": self.synth.checkpoint,
                    "sample_rate": OUT_SAMPLE_RATE,
                    "native_sample_rate": NATIVE_SAMPLE_RATE,
                    "expressions": list(EXPRESSION_CAPTIONS),
                    # 参照音声なしの声デザインが使えるチェックポイントかどうか
                    "voice_design": bool(
                        getattr(self.synth.runtime.model_cfg, "use_caption_condition", False)
                    ),
                },
            )
            return
        if path == "/speakers":
            self._send_json(200, [s.to_public() for s in self.synth.speakers.values()])
            return
        if path.startswith("/jobs/"):
            try:
                self._send_json(200, self.synth.get_job(path[len("/jobs/"):]).to_public())
            except ValueError as e:
                self._send_json(404, {"error": str(e)})
            return
        self._send_json(404, {"error": f"知らないパスです: {path}"})

    def do_PATCH(self):  # noqa: N802
        path = self._path()
        if path.startswith("/speakers/"):
            params = self._read_json()
            if params is None:
                return
            try:
                # _path() が既に unquote 済みなので、ここでは切り出すだけ（DELETE と同じ作法）
                self._send_json(
                    200,
                    self.synth.rename_speaker(path[len("/speakers/"):], params.get("name", "")),
                )
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
            except Exception as e:
                log.exception("話者の名前変更に失敗")
                self._send_json(500, {"error": f"話者の名前変更に失敗しました: {e}"})
            return
        self._send_json(404, {"error": f"知らないパスです: {path}"})

    def do_DELETE(self):  # noqa: N802
        path = self._path()
        if path.startswith("/speakers/"):
            try:
                self._send_json(200, self.synth.delete_speaker(path[len("/speakers/"):]))
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
            except Exception as e:
                log.exception("話者の削除に失敗")
                self._send_json(500, {"error": f"話者の削除に失敗しました: {e}"})
            return
        self._send_json(404, {"error": f"知らないパスです: {path}"})

    def do_POST(self):  # noqa: N802
        path = self._path()
        if path == "/synthesis":
            params = self._read_json()
            if params is None:
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
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(wav_bytes)))
            self.end_headers()
            self.wfile.write(wav_bytes)
            return

        if path == "/speakers":
            params = self._read_json()
            if params is None:
                return
            try:
                self._send_json(200, self.synth.add_speaker(params))
            except (ValueError, RuntimeError) as e:
                self._send_json(400, {"error": str(e)})
            except Exception as e:
                log.exception("話者の追加に失敗")
                self._send_json(500, {"error": f"話者の追加に失敗しました: {e}"})
            return

        if path == "/design":
            params = self._read_json()
            if params is None:
                return
            try:
                wav_bytes, seed = self.synth.design_preview(params)
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return
            except Exception as e:  # 落とさずに理由を返す
                log.exception("声デザインの試し聞きに失敗")
                self._send_json(500, {"error": f"声デザインに失敗しました: {e}"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Access-Control-Allow-Origin", "*")
            # 使った種を返す。これを登録に渡すと同じ声がもう一度出る
            self.send_header("X-Irodori-Seed", str(seed))
            self.send_header("Access-Control-Expose-Headers", "X-Irodori-Seed")
            self.send_header("Content-Length", str(len(wav_bytes)))
            self.end_headers()
            self.wfile.write(wav_bytes)
            return

        if path == "/design/speakers":
            params = self._read_json()
            if params is None:
                return
            try:
                self._send_json(200, self.synth.start_design_speaker(params).to_public())
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
            except Exception as e:
                log.exception("声デザイン登録の開始に失敗")
                self._send_json(500, {"error": f"声デザイン登録に失敗しました: {e}"})
            return

        if path == "/render":
            params = self._read_json()
            if params is None:
                return
            try:
                self._send_json(200, self.synth.start_render(params).to_public())
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
            except Exception as e:
                log.exception("レンダー開始に失敗")
                self._send_json(500, {"error": f"レンダー開始に失敗しました: {e}"})
            return

        m = re.fullmatch(r"/jobs/([0-9a-f]+)/cancel", path)
        if m:
            try:
                job = self.synth.get_job(m.group(1))
                job.cancel.set()
                self._send_json(200, job.to_public())
            except ValueError as e:
                self._send_json(404, {"error": str(e)})
            return

        self._send_json(404, {"error": f"知らないパスです: {path}"})


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
