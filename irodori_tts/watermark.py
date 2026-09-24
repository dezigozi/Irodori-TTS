from __future__ import annotations

import logging
from collections.abc import Iterable

import torch

logger = logging.getLogger(__name__)

IRODORI_WATERMARK_PAYLOAD = (73, 82, 68, 84, 83)  # "IRDTS"


def _as_single_channel_vector(audio: torch.Tensor) -> torch.Tensor | None:
    squeezed = audio.detach().float().squeeze()
    if squeezed.ndim == 0 or squeezed.numel() == 0:
        return None
    if squeezed.ndim == 1:
        return squeezed
    return squeezed.reshape(-1)


def _match_original_rank(audio: torch.Tensor, *, reference: torch.Tensor) -> torch.Tensor:
    if reference.ndim == 2:
        return audio.reshape(1, -1)
    return audio.reshape(-1)


class _BatchNorm2dWithoutGraphCache(torch.nn.Module):
    """BatchNorm2d と同じ計算を、MPS の形ごとのグラフキャッシュを通さずにやる。

    MPS の batch_norm は入力の形ごとに MPSGraph を作ってキャッシュし、1形あたり
    100〜500MB を抱えたまま返さへん（torch.mps.empty_cache() でも消えへん。PyTorch 2.10 で実測）。
    透かしは音の長さ＝毎回ちがう形で呼ばれるので、常駐サーバやとチャンクごとに膨らみ続けてた。
    var_mean と四則演算に分けたら形が変わっても増えへん（数値差は 1e-6 程度）。

    SilentCipher は eval() を呼ばへんので BN は train モード（その入力のバッチ統計で正規化）のまま
    使われとる。ここでも train なら入力の統計、eval なら running 統計、と元の挙動をそのまま真似る。
    （train モードの running 統計の更新は出力に効かへんのでやらへん）
    """

    def __init__(self, bn: torch.nn.BatchNorm2d) -> None:
        super().__init__()
        self.bn = bn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bn = self.bn
        if bn.training or bn.running_mean is None or bn.running_var is None:
            var, mean = torch.var_mean(x, dim=(0, 2, 3), unbiased=False, keepdim=True)
        else:
            mean = bn.running_mean.view(1, -1, 1, 1)
            var = bn.running_var.view(1, -1, 1, 1)
        y = (x - mean) * torch.rsqrt(var + bn.eps)
        if bn.affine:
            y = y * bn.weight.view(1, -1, 1, 1) + bn.bias.view(1, -1, 1, 1)
        return y


def _swap_batchnorm2d(module: torch.nn.Module) -> int:
    """module 配下の BatchNorm2d を差し替えて、差し替えた数を返す。"""
    swapped = 0
    for name, child in module.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            setattr(module, name, _BatchNorm2dWithoutGraphCache(child))
            swapped += 1
        else:
            swapped += _swap_batchnorm2d(child)
    return swapped


class SilentCipherWatermarker:
    def __init__(self, *, device: str, model_type: str = "44.1k") -> None:
        self.model = self._load_backend(device=device, model_type=model_type)
        if self.model is not None and torch.device(device).type == "mps":
            # 透かしを入れる経路（enc_c / dec_c）だけ差し替える。検出側（dec_m）はここでは使わへん
            swapped = sum(
                _swap_batchnorm2d(m)
                for m in (getattr(self.model, "enc_c", None), getattr(self.model, "dec_c", None))
                if isinstance(m, torch.nn.Module)
            )
            logger.info("SilentCipher: MPS 用に BatchNorm2d を %d 個差し替えました", swapped)

    @staticmethod
    def _load_backend(*, device: str, model_type: str):
        try:
            import silentcipher
        except ImportError:
            logger.warning(
                "SilentCipher package is unavailable; generated audio will not be watermarked."
            )
            return None

        try:
            return silentcipher.get_model(model_type=model_type, device=device)
        except Exception as exc:
            logger.warning(
                "SilentCipher model could not be loaded (%s); generated audio will not be "
                "watermarked.",
                exc,
            )
            return None

    @property
    def ready(self) -> bool:
        return self.model is not None

    def encode_one(
        self,
        audio: torch.Tensor,
        *,
        sample_rate: int,
        payload: Iterable[int] = IRODORI_WATERMARK_PAYLOAD,
    ) -> torch.Tensor:
        if self.model is None:
            return audio

        vector = _as_single_channel_vector(audio)
        if vector is None:
            return audio

        encoded, _ = self.model.encode_wav(
            vector.to(self.model.device),
            int(sample_rate),
            list(payload),
            calc_sdr=False,
        )
        encoded_audio = torch.as_tensor(encoded, dtype=torch.float32, device="cpu")
        return _match_original_rank(encoded_audio, reference=audio)

    def encode_batch(self, audios: list[torch.Tensor], *, sample_rate: int) -> list[torch.Tensor]:
        if self.model is None:
            return audios
        return [self.encode_one(audio, sample_rate=sample_rate) for audio in audios]
