from functools import partial
from typing import Any, Dict, Optional, Union

import torch
from torch import nn

from .functional import imdct, mdct
from .windows import kaiser_bessel_derived, vorbis


class MDCT(nn.Module):
    """
    Module to compute the Modified Discrete Cosine Transform (MDCT) of a waveform.

    Parameters
    ----------
    win_length : int
        The length of the window.
    window_fn : callable, optional
        A function to generate the window, by default vorbis.
    window_kwargs : dict, optional
        Additional keyword arguments to pass to the window function, by default {}.
    center : bool, optional
        If True, pad the waveform on both sides with half the window length, by default True.

    Attributes
    ----------
    window : torch.Tensor
        The window tensor.
    mdct : callable
        The MDCT function with the specified center parameter.

    Methods
    -------
    forward(waveform: torch.Tensor) -> torch.Tensor
        Compute the MDCT of the input waveform.
    """

    def __init__(
        self,
        win_length: int,
        window_fn: Union[kaiser_bessel_derived, vorbis] = vorbis,
        window_kwargs: Dict[str, Any] = {},
        center: bool = True,
    ) -> None:
        super().__init__()

        self.register_buffer("window", window_fn(win_length, **window_kwargs))
        self.mdct = partial(mdct, center=center)

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
            MDCT spectrogram of the input waveform of shape (..., win_length // 2, n_frames).
        """
        return self.mdct(waveform, self.window)


class IMDCT(nn.Module):
    """
    Module to compute the inverse Modified Discrete Cosine Transform (iMDCT) of a spectrogram.

    Parameters
    ----------
    win_length : int
        The length of the window.
    window_fn : callable, optional
        A function to generate the window, by default vorbis.
    window_kwargs : dict, optional
        Additional keyword arguments to pass to the window function, by default {}.
    center : bool, optional
        If True, remove the padding added during MDCT, by default True.

    Attributes
    ----------
    window : torch.Tensor
        The window tensor.
    imdct : callable
        The iMDCT function with the specified center parameter.

    Methods
    -------
    forward(spectrogram: torch.Tensor) -> torch.Tensor
        Compute the iMDCT of the input spectrogram.
    """

    def __init__(
        self,
        win_length: int,
        window_fn: Union[kaiser_bessel_derived, vorbis] = vorbis,
        window_kwargs: Dict[str, Any] = {},
        center: bool = True,
    ) -> None:
        super().__init__()

        self.register_buffer("window", window_fn(win_length, **window_kwargs))
        self.imdct = partial(imdct, center=center)

    def forward(
        self, spectrogram: torch.Tensor, length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute the iMDCT of the input spectrogram.

        Parameters
        ----------
        spectrogram : torch.Tensor
            Input MDCT spectrogram tensor of shape (..., win_length // 2, n_frames).

        length : int, optional
            The length of the output waveform, by default None.

        Returns
        -------
        torch.Tensor
            Reconstructed waveform tensor of shape (..., n_samples).
        """
        length = length or -1  # By default, return the full waveform
        return self.imdct(spectrogram, self.window)[..., :length]
