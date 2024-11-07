import torch


def kaiser_bessel_derived(
    win_length: int,
    beta: float = 12.0,
    *,
    dtype: torch.dtype = None,
    device: torch.device = None
) -> torch.Tensor:
    """
    Generate a Kaiser-Bessel derived window.

    Parameters
    ----------
    win_length : int
        The length of the window.
    beta : float, optional
        The beta parameter for the Kaiser window, by default 12.0.
    dtype : torch.dtype, optional
        The desired data type of returned tensor, by default None.
    device : torch.device, optional
        The desired device of returned tensor, by default None.

    Returns
    -------
    torch.Tensor
        The generated Kaiser-Bessel derived window of shape (win_length,).
    """

    half_w_length = win_length // 2
    kaiser_w = torch.kaiser_window(
        half_w_length + 1, True, beta, dtype=dtype, device=device
    )
    kaiser_w_csum = torch.cumsum(kaiser_w, dim=-1)
    half_w = torch.sqrt(kaiser_w_csum[:-1] / kaiser_w_csum[-1])
    w = torch.cat((half_w, torch.flip(half_w, dims=(0,))), axis=0)

    return w


def vorbis(
    win_length: int, *, dtype: torch.dtype = None, device: torch.device = None
) -> torch.Tensor:
    """
    Generate a Vorbis window.

    Parameters
    ----------
    win_length : int
        The length of the window.
    dtype : torch.dtype, optional
        The desired data type of returned tensor, by default None.
    device : torch.device, optional
        The desired device of returned tensor, by default None.

    Returns
    -------
    torch.Tensor
        The generated Vorbis window of shape (win_length,).
    """

    arg = torch.arange(win_length, dtype=dtype, device=device) + 0.5
    w = torch.sin(
        torch.pi / 2.0 * torch.pow(torch.sin(torch.pi / win_length * arg), 2.0)
    )

    return w
