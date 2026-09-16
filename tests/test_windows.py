import pytest
import torch

from torch_mdct.functional import imdct, mdct
from torch_mdct.windows import kaiser_bessel_derived, vorbis

WINDOW_FNS = [vorbis, kaiser_bessel_derived]
WIN_LENGTHS = [64, 256, 1024]


@pytest.mark.parametrize("window_fn", WINDOW_FNS)
@pytest.mark.parametrize("win_length", WIN_LENGTHS)
def test_princen_bradley_condition(window_fn, win_length):
    """Every MDCT/IMDCT window must satisfy the Princen-Bradley time-domain
    aliasing cancellation condition

        w[n]**2 + w[n + win_length // 2]**2 == 1   for n in [0, win_length // 2)

    kaiser_bessel_derived previously violated this by as much as 0.084 at
    win_length=64 (and still 0.005 at win_length=1024) because its
    half-window was built from an asymmetric (periodic=True) Kaiser
    window rather than a symmetric one, which the cumulative-sum KBD
    construction requires.
    """
    n = win_length // 2
    w = window_fn(win_length, dtype=torch.float64)
    lhs = w[:n] ** 2 + w[n:] ** 2
    max_dev = (lhs - 1).abs().max().item()
    assert max_dev < 1e-10, f"Princen-Bradley deviation {max_dev} too large"


@pytest.mark.parametrize("window_fn", WINDOW_FNS)
@pytest.mark.parametrize("win_length", WIN_LENGTHS)
def test_mdct_imdct_perfect_reconstruction(window_fn, win_length):
    """Regression test for the kaiser_bessel_derived perfect-reconstruction
    bug: with the buggy window, round-trip error was as large as 0.33 for
    a unit-scale signal (not a small numerical artifact). Tolerance here
    is set to the library's own float32-internal precision ceiling
    (functional.mdct/imdct compute twiddle factors without forwarding the
    input dtype), not machine epsilon, so this isolates the window-symmetry
    bug rather than that separate, much smaller precision limitation.
    """
    torch.manual_seed(0)
    n_samples = win_length * 6 + 13  # deliberately not a multiple of win_length
    x = torch.randn(1, n_samples, dtype=torch.float64)
    window = window_fn(win_length, dtype=x.dtype)

    spec = mdct(x, window)
    x_hat = imdct(spec, window, n_samples=n_samples)

    err = (x_hat - x).abs().max().item()
    assert err < 2e-4, f"reconstruction error {err} too large"
