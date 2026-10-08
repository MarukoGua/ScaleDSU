import numpy as np
import torch
from sklearn import preprocessing


def no_preprocessing(X):
    return X

def feature_standardization(X):
    scaler = preprocessing.StandardScaler()
    X_processed = scaler.fit_transform(X)
    return X_processed

def shift(traces, max_delay=50):
    sample_len = traces.shape[1]
    shifted_traces = []
    for i in range(traces.shape[0]):
        random_int = np.random.randint(0, max_delay, size=1)
        trace_i = traces[i]
        paddings = np.zeros(abs(random_int))
        trace_i = list(paddings) + list(trace_i)
        trace_i = trace_i[:sample_len]
        shifted_traces.append(trace_i)

    shifted_traces = np.array(shifted_traces)
    return shifted_traces


def notch_many_fft(x, fs, freqs, Q, taper_ratio, atten_db, mix):
    """
    Narrow-band noise removal using smooth frequency-domain notches.

    Args:
        x: 1D (T,) or 2D (N, T) real signal. torch tensor or numpy array.
        fs: Sample rate in Hz.
        freqs: Iterable of center frequencies to notch (Hz).
        Q: Quality factor; -3 dB bandwidth BW ≈ f0 / Q. Larger Q => narrower notch.
        taper_ratio: Cosine taper fraction of the half-band. 0 = brick-wall, 0.5 = pure cosine.
        atten_db: Notch depth in dB (amplitude domain). 60 dB means ~1e-3 gain at center.
        mix: Wet/dry mix. y = mix * filtered + (1-mix) * original.
        device: Torch device. If None, inferred from x or defaults to CPU.

    Returns:
        y : filtered signal with same shape/type as input.
    """
    # --- Normalize input to torch tensor (float32) and batch shape (B, T) ---
    x = torch.from_numpy(np.asarray(x)).float()
    assert isinstance(x, torch.Tensor)

    B, T = x.shape
    n_r = T // 2 + 1

    # --- rFFT and frequency bins ---
    X = torch.fft.rfft(x, dim=-1)  # (B, n_r)
    # rFFT bin frequencies: f_k = k * fs / T
    f_bins = torch.arange(n_r, dtype=x.dtype) * (fs / T)

    # --- Build smooth multiplicative mask: start from ones (pass-through) ---
    mask = torch.ones(n_r, dtype=x.dtype)

    # Precompute center attenuation (amplitude gain)
    att = 10.0 ** (-atten_db / 20.0)

    for f0 in freqs:
        # Skip invalid or out-of-band centers
        if not (0.0 < f0 < fs * 0.5):
            continue

        # Half-bandwidth by Q: H = BW/2 ≈ f0 / (2Q)
        H = float(f0) / (2.0 * float(Q))
        if H <= 0:
            continue

        # Split half-band into a flat "core" and cosine "skirts"
        taper = max(taper_ratio * H, 0.0)
        core  = max(H - taper, 0.0)

        # Distance from the center frequency
        delta = (f_bins - f0).abs()

        if taper > 0:
            # Smooth gain profile:
            #   delta <= core: gain = att
            #   core < delta < core+taper: cosine ramp from att -> 1
            #   delta >= core+taper: gain = 1
            t = ((delta - core) / taper).clamp(0.0, 1.0)          # 0..1
            r = 0.5 - 0.5 * torch.cos(torch.pi * t)               # 0..1 (cosine ease)
            gain = att + (1.0 - att) * r                          # att..1
        else:
            # Brick-wall notch within |delta| <= H
            gain = torch.where(delta <= H, torch.tensor(att, dtype=x.dtype),
                                           torch.tensor(1.0, dtype=x.dtype))

        mask = mask * gain  # combine multiple notches multiplicatively

    # --- Apply mask and inverse rFFT ---
    Xf = X * mask[None, :]
    y = torch.fft.irfft(Xf, n=T, dim=-1)

    # Wet/dry mix
    if mix != 1.0:
        y = mix * y + (1.0 - mix) * x

    # Restore original shape/type
    y = y.squeeze(0) if B == 1 else y
    y = y.cpu().numpy()

    return y