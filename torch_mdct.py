import math
from functools import partial
from typing import Any, Callable, Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def mdct(waveform: Tensor, window: Tensor, center: bool = True) -> Tensor:
    n_samples = waveform.shape[-1]
    win_length = window.shape[-1]
    hop_length = win_length // 2

    # Flatten the input tensor
    shape = waveform.shape
    waveform = waveform.flatten(end_dim=-2)

    # Center the waveform by padding on both sides
    n_frames = int(math.ceil(n_samples / hop_length)) + 1
    if center:
        waveform = F.pad(waveform,
                         (hop_length, (n_frames + 1) * hop_length - n_samples),
                         mode="constant")

    # Prepare pre&post processing arrays
    pre_twiddle = torch.exp(
        -1j * torch.pi / win_length *
        torch.arange(0, win_length, device=waveform.device))
    post_twiddle = torch.exp(
        -1j * torch.pi / win_length * (win_length / 2 + 1) *
        torch.arange(0.5, win_length / 2 + 0.5, device=waveform.device))

    # Convolve the waveform with the window and fft matrices
    specgram = waveform.unfold(dimension=-1, size=win_length, step=hop_length)
    specgram = torch.einsum("...kj,j->...kj", specgram, window)
    specgram = torch.einsum("...kj,j->...kj", specgram, pre_twiddle)
    specgram = torch.fft.fft(specgram, dim=-1)
    specgram = torch.einsum("...jk,k->...kj", specgram[..., :win_length // 2],
                            post_twiddle)
    specgram = torch.real(specgram)

    # Unflatten the output
    specgram = specgram.reshape(shape[:-1] + specgram.shape[-2:])

    return specgram


def imdct(specgram: Tensor, window: Tensor, center: bool = True) -> Tensor:
    win_length = window.shape[-1]
    hop_length = win_length // 2
    n_freqs, n_frames = specgram.shape[-2:]
    n_samples = hop_length * (n_frames + 1)

    # Flatten the input tensor
    shape = specgram.shape
    specgram = specgram.flatten(end_dim=-3)

    # Prepare pre&post processing arrays
    pre_twiddle = torch.exp(-1j * torch.pi / (2 * n_freqs) * (n_freqs + 1) *
                            torch.arange(n_freqs, device=specgram.device))
    post_twiddle = torch.exp(-1j * torch.pi / (2 * n_freqs) *
                             torch.arange(0.5 + n_freqs / 2,
                                          2 * n_freqs + n_freqs / 2 + 0.5,
                                          device=specgram.device)) / n_freqs

    # Apply fft and the window
    specgram = torch.einsum("...jk,j->...jk", specgram, pre_twiddle)
    specgram = torch.fft.fft(specgram, n=2 * n_freqs, dim=1)
    specgram = torch.einsum("...jk,j->...jk", specgram, post_twiddle)
    specgram = torch.real(specgram)
    specgram = 2 * torch.einsum("...jk,j->...jk", specgram, window)

    # Recover the waveform with the time-domain aliasing cancelling principle
    waveform = F.fold(specgram, (1, n_samples),
                      kernel_size=(1, win_length),
                      stride=(1, hop_length))

    # Remove padding
    if center:
        waveform = waveform[..., hop_length:-hop_length]

    # Unflatten the output
    waveform = waveform.reshape((*shape[:-2], -1))

    return waveform


def kaiser_bessel_derived_window(win_length: int,
                                 beta: float = 12.0,
                                 *,
                                 dtype: torch.dtype = None,
                                 device: torch.device = None) -> Tensor:
    half_w_length = win_length // 2
    kaiser_w = torch.kaiser_window(half_w_length + 1,
                                   True,
                                   beta,
                                   dtype=dtype,
                                   device=device)
    kaiser_w_csum = torch.cumsum(kaiser_w, dim=-1)
    half_w = torch.sqrt(kaiser_w_csum[:-1] / kaiser_w_csum[-1])
    w = torch.cat((half_w, torch.flip(half_w, dims=(0, ))), axis=0)

    return w


def vorbis_window(win_length: int,
                  *,
                  dtype: torch.dtype = None,
                  device: torch.device = None) -> Tensor:
    arg = torch.arange(win_length, dtype=dtype, device=device) + 0.5
    w = torch.sin(torch.pi / 2.0 *
                  torch.pow(torch.sin(torch.pi / win_length * arg), 2.0))

    return w


class MDCT(nn.Module):

    def __init__(self,
                 win_length: int,
                 window_fn: Callable[..., Tensor] = vorbis_window,
                 wkwargs: Optional[Dict[str, Any]] = None,
                 center: bool = True) -> None:
        super().__init__()

        window = window_fn(win_length) if wkwargs is None else window_fn(
            win_length, **wkwargs)
        self.register_buffer("window", window)
        self.mdct = partial(mdct, center=center)

    def forward(self, waveform: Tensor) -> Tensor:
        return self.mdct(waveform, self.window)


class InverseMDCT(nn.Module):

    def __init__(self,
                 win_length: int,
                 window_fn: Callable[..., Tensor] = vorbis_window,
                 wkwargs: Optional[Dict[str, Any]] = None,
                 center: bool = True) -> None:
        super().__init__()

        window = window_fn(win_length) if wkwargs is None else window_fn(
            win_length, **wkwargs)
        self.register_buffer("window", window)
        self.imdct = partial(imdct, center=center)

    def forward(self,
                specgram: Tensor,
                length: Optional[int] = None) -> Tensor:
        if length is None:
            return self.imdct(specgram, self.window)

        return self.imdct(specgram, self.window)[..., :length]
