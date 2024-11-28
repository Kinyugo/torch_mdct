import math

import torch
from torch.nn import functional as F


def mdct(
    waveform: torch.Tensor, window: torch.Tensor, center: bool = True
) -> torch.Tensor:
    """
    Compute the Modified Discrete Cosine Transform (MDCT) of a waveform.

    Parameters
    ----------
    waveform : torch.Tensor
        Input waveform tensor of shape (..., n_samples).
    window : torch.Tensor
        Window tensor of shape (win_length,).
    center : bool, optional
        If True, pad the waveform on both sides with half the window length, by default True.

    Returns
    -------
    torch.Tensor
        MDCT spectrogram of the input waveform of shape (..., win_length // 2, n_frames).
    """
    n_samples = waveform.shape[-1]
    win_length = window.shape[-1]
    hop_length = win_length // 2

    # Flatten the input tensor
    shape = waveform.shape
    waveform = waveform.flatten(end_dim=-2)

    # Center the waveform by padding on both sides
    n_frames = int(math.ceil(n_samples / hop_length)) + 1
    if center:
        waveform = F.pad(
            waveform,
            (hop_length, (n_frames + 1) * hop_length - n_samples),
            mode="constant",
        )

    # Prepare pre&post processing arrays
    pre_twiddle = torch.exp(
        -1j
        * torch.pi
        / win_length
        * torch.arange(0, win_length, device=waveform.device)
    )
    post_twiddle = torch.exp(
        -1j
        * torch.pi
        / win_length
        * (win_length / 2 + 1)
        * torch.arange(0.5, win_length / 2 + 0.5, device=waveform.device)
    )

    # Convolve the waveform with the window and fft matrices
    spectrogram = waveform.unfold(dimension=-1, size=win_length, step=hop_length)
    spectrogram = torch.einsum("...kj,j->...kj", spectrogram, window)
    spectrogram = torch.einsum("...kj,j->...kj", spectrogram, pre_twiddle)
    spectrogram = torch.fft.fft(spectrogram, dim=-1)
    spectrogram = torch.einsum(
        "...jk,k->...kj", spectrogram[..., : win_length // 2], post_twiddle
    )
    spectrogram = torch.real(spectrogram)

    # Unflatten the output
    spectrogram = spectrogram.reshape(shape[:-1] + spectrogram.shape[-2:])

    # Scale the output
    spectrogram = (1.0 / math.sqrt(4.0 * win_length)) * spectrogram

    return spectrogram


def imdct(
    spectrogram: torch.Tensor, window: torch.Tensor, center: bool = True
) -> torch.Tensor:
    """
    Compute the inverse Modified Discrete Cosine Transform (iMDCT) of a spectrogram.

    Parameters
    ----------
    spectrogram : torch.Tensor
        Input MDCT spectrogram tensor of shape (..., win_length // 2, n_frames).
    window : torch.Tensor
        Window tensor of shape (win_length,).
    center : bool, optional
        If True, remove the padding added during MDCT, by default True.

    Returns
    -------
    torch.Tensor
        Reconstructed waveform tensor of shape (..., n_samples).
    """
    win_length = window.shape[-1]
    hop_length = win_length // 2
    n_freqs, n_frames = spectrogram.shape[-2:]
    n_samples = hop_length * (n_frames + 1)

    # Unscale the input
    spectrogram = math.sqrt(4.0 * win_length) * spectrogram

    # Flatten the input tensor
    shape = spectrogram.shape
    spectrogram = spectrogram.flatten(end_dim=-3)

    # Prepare pre&post processing arrays
    pre_twiddle = torch.exp(
        -1j
        * torch.pi
        / (2 * n_freqs)
        * (n_freqs + 1)
        * torch.arange(n_freqs, device=spectrogram.device)
    )
    post_twiddle = (
        torch.exp(
            -1j
            * torch.pi
            / (2 * n_freqs)
            * torch.arange(
                0.5 + n_freqs / 2,
                2 * n_freqs + n_freqs / 2 + 0.5,
                device=spectrogram.device,
            )
        )
        / n_freqs
    )

    # Apply fft and the window
    spectrogram = torch.einsum("...jk,j->...jk", spectrogram, pre_twiddle)
    spectrogram = torch.fft.fft(spectrogram, n=2 * n_freqs, dim=1)
    spectrogram = torch.einsum("...jk,j->...jk", spectrogram, post_twiddle)
    spectrogram = torch.real(spectrogram)
    spectrogram = 2 * torch.einsum("...jk,j->...jk", spectrogram, window)

    # Recover the waveform with the time-domain aliasing cancelling principle
    waveform = F.fold(
        spectrogram, (1, n_samples), kernel_size=(1, win_length), stride=(1, hop_length)
    )

    # Remove padding
    if center:
        waveform = waveform[..., hop_length:-hop_length]

    # Unflatten the output
    waveform = waveform.reshape((*shape[:-2], -1))

    return waveform
