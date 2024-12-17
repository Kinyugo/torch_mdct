# torch_mdct

A PyTorch implementation of the Modified Discrete Cosine Transform (MDCT) and its inverse for audio processing.

## Installation 

```bash
pip install torch_mdct
```

## Usage

```python
import torchaudio
from torch_mdct import IMDCT, MDCT, kaiser_bessel_derived, vorbis

# Load a sample waveform 
waveform, sample_rate = torchaudio.load("/path/to/audio.file")

# Initialize the mdct and imdct transforms
mdct = MDCT(win_length=1024, window_fn=vorbis, window_kwargs=None, center=True)
imdct = IMDCT(win_length=1024, window_fn=vorbis, window_kwargs=None, center=True)

# Transform waveform into mdct spectrogram
spectrogram = mdct(waveform)

# Transform spectrogram back to audio 
reconst_waveform = imdct(spectrogram)

# Compute the differences
print(f"L1: {(waveform - reconst_waveform).abs().mean()}")
```

## References 
[[1]](https://github.com/zafarrafii/Zaf-Python) Zaf-Python: Zafar's Audio Functions in **Python** for audio signal analysis.

[[2]](https://github.com/nils-werner/mdct) MDCT: A fast MDCT implementation using SciPy and FFTs.