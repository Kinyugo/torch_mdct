from typing import Any, Callable, Dict, Optional

import torch
from torch import nn

from .functional import imdct, mdct
from .windows import vorbis


def create_window(
    win_length: int,
    window_fn: Callable[..., torch.Tensor],
    window_kwargs: Optional[Dict[str, Any]],
) -> torch.Tensor:
    """
    Utility function to create a window tensor.

    Parameters
    ----------
    win_length : int
        The length of the window.
    window_fn : callable
        Function to generate the window.
    window_kwargs : dict, optional
        Additional keyword arguments to pass to the window function.

    Returns
    -------
    torch.Tensor
        Precomputed window tensor.
    """
    return window_fn(win_length, **(window_kwargs or {}))


class MDCT(nn.Module):
    """
    Module to compute the Modified Discrete Cosine Transform (MDCT) of a waveform.

    Parameters
    ----------
    win_length : int
        The length of the window.
    window_fn : callable, default=vorbis
        A function to generate the window, by default vorbis. kaiser_bessel_derived is also available.
    window_kwargs : dict, optional
        Additional keyword arguments to pass to the window function, by default None.
    center : bool, default=True
        If True, pad the waveform on both sides with half the window length, by default True.

    Attributes
    ----------
    window : torch.Tensor
        The window tensor.

    Methods
    -------
    forward(waveform: torch.Tensor) -> torch.Tensor
        Compute the MDCT of the input waveform.

    Examples
    --------
    >>> waveform = torch.rand(2, 44100) # (channels, n_samples)
    >>> mdct = MDCT(win_length=1024)
    >>> spectrogram = mdct(waveform)
    >>> print(spectrogram.shape)  # (2, 512, 89)
    """

    def __init__(
        self,
        win_length: int,
        window_fn: Callable[..., torch.Tensor] = vorbis,
        window_kwargs: Optional[Dict[str, Any]] = None,
        center: bool = True,
    ) -> None:
        super().__init__()

        self.win_length = win_length
        self.window_fn = window_fn
        self.window_kwargs = window_kwargs or {}
        self.center = center

        self.register_buffer(
            "window", create_window(self.win_length, self.window_fn, self.window_kwargs)
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute the MDCT of the input waveform.

        Parameters
        ----------
        waveform : torch.Tensor
            Input waveform tensor of shape (..., n_samples).

        Returns
        -------
        torch.Tensor
            MDCT spectrogram of shape (..., win_length // 2, n_frames).
        """
        return mdct(waveform, self.window, center=self.center)


class IMDCT(nn.Module):
    """
    Module to compute the inverse Modified Discrete Cosine Transform (iMDCT) of a spectrogram.

    Parameters
    ----------
    win_length : int
        The length of the window.
    window_fn : callable, default=vorbis
        A function to generate the window, by default vorbis. kaiser_bessel_derived is also available.
    window_kwargs : dict, optional
        Additional keyword arguments to pass to the window function, by default None.
    center : bool, default=True
        If True, pad the waveform on both sides with half the window length, by default True.

    Attributes
    ----------
    window : torch.Tensor
        The window tensor.

    Methods
    -------
    forward(spectrogram: torch.Tensor, *, n_samples: Optional[int] = None) -> torch.Tensor
        Compute the iMDCT of the input spectrogram.

    Examples
    --------
    >>> spectrogram = torch.rand(2, 512, 89) # (channels, win_length // 2, n_frames)
    >>> imdct = IMDCT(win_length=1024)
    >>> waveform = imdct(spectrogram, n_samples=44100) # (2, 44100)
    """

    def __init__(
        self,
        win_length: int,
        window_fn: Callable[..., torch.Tensor] = vorbis,
        window_kwargs: Optional[Dict[str, Any]] = None,
        center: bool = True,
    ) -> None:
        super().__init__()

        # Save parameters for introspection or serialization
        self.win_length = win_length
        self.window_fn = window_fn
        self.window_kwargs = window_kwargs or {}
        self.center = center

        # Register window tensor
        self.register_buffer(
            "window", create_window(self.win_length, self.window_fn, self.window_kwargs)
        )

    def forward(
        self, spectrogram: torch.Tensor, *, n_samples: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute the iMDCT of the input spectrogram.

        Parameters
        ----------
        spectrogram : torch.Tensor
            Input MDCT spectrogram tensor of shape (..., win_length // 2, n_frames).

        n_samples : int, optional
            Desired length of the output waveform.

        Returns
        -------
        torch.Tensor
            Reconstructed waveform tensor of shape (..., n_samples).
        """
        return imdct(spectrogram, self.window, center=self.center, n_samples=n_samples)
