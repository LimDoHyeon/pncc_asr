from typing import Tuple
import numpy as np
import torch
import torchaudio
from scipy import signal  # for Gammatone design only (CPU)


def load_audio(path: str, target_sr: int, device: torch.device = 'cpu') -> Tuple[torch.Tensor, int]:
    """Load mono wav, resample, move to *device* (shape [T])."""
    wav, sr = torchaudio.load(path)  # [C, T]
    wav = wav[0] if wav.shape[0] > 1 else wav.squeeze(0)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        sr = target_sr
    return wav.to(device), sr


def erb_scale_freqs(f_min: float, f_max: float, n: int) -> np.ndarray:
    """ERB‑spaced center‑frequencies (Glasberg & Moore)."""

    # ERB number formula: ERB(f)=24.7*(4.37e-3*f+1)
    def hz_to_erb(f):
        return 21.4 * np.log10(4.37e-3 * f + 1)

    def erb_to_hz(erb):
        return (10 ** (erb / 21.4) - 1) / 4.37e-3

    erb_low, erb_high = hz_to_erb(f_min), hz_to_erb(f_max)
    erb_points = np.linspace(erb_low, erb_high, n)
    return erb_to_hz(erb_points)


def build_gammatone_bank(sr, n_fft, n_ch,
                         f_min: int, f_max: int) -> torch.Tensor:
    """Return torch tensor [n_ch, n_fft//2+1] of |H_l(e^{jω_k})|^2.

    * Each channel is area‑normalised: Σ_k |H_l|² = 1.
    * Magnitude squared below 0.5 % peak (−46 dB) → set to 0.
    """
    nyq = sr * 0.5
    if f_max >= nyq:
        f_max = nyq * 0.999
    cfs = erb_scale_freqs(f_min, f_max, n_ch)
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    resp = np.empty((n_ch, freqs.size), dtype=np.float32)

    for i, cf in enumerate(cfs):
        # IIR gammatone design, order=4
        b, a = signal.gammatone(cf, "iir", fs=sr)
        w, h = signal.freqz(b, a, worN=freqs, fs=sr)
        mag2 = np.abs(h) ** 2
        # area normalisation Σ|H|² = 1 on discrete grid
        mag2 /= mag2.sum()
        # clip values below −46 dB of max
        threshold = mag2.max() * 0.005  # 0.5 %
        mag2[mag2 < threshold] = 0.0
        resp[i] = mag2
    return torch.tensor(resp)  # CPU tensor; caller moves to device


def pre_emphasis(x: torch.Tensor, coeff: float = 0.97) -> torch.Tensor:
    # y = torch.empty_like(x)
    # y[0] = x[0]
    # y[1:] = x[1:] - coeff * x[:-1]
    # return y
    if x.dim() == 1:
        y = torch.empty_like(x)
        y[0] = x[0]
        y[1:] = x[1:] - coeff * x[:-1]
        return y
    elif x.dim() == 2:
        y = torch.empty_like(x)
        y[:, 0] = x[:, 0]
        y[:, 1:] = x[:, 1:] - coeff * x[:, :-1]
        return y
    else:
        raise ValueError(f"pre_emphasis expects 1D or 2D tensor, got {x.shape}")


def stft_power(x: torch.Tensor,
               n_fft,
               hop_length,
               win_length) -> torch.Tensor:
    window = torch.hamming_window(win_length, device=x.device)
    X = torch.stft(x, n_fft, hop_length=hop_length, win_length=win_length,
                   window=window, return_complex=True)
    return X.abs().pow(2)  # [B, F, T]


def apply_filterbank(power_spec: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    return torch.matmul(H, power_spec)


def medium_time_avg(P: torch.Tensor, M: int = 2) -> torch.Tensor:
    """5‑frame (~65 ms) moving average over time axis (dim=1)."""
    if P.dim() == 2:  # (L,T) → (1,L,T)
        P = P.unsqueeze(0)
    elif P.dim() != 3:
        raise ValueError(f"Expected 2D/3D tensor, got {P.shape}")
    B, L, T = P.shape
    kernel = torch.ones(L, 1, 2 * M + 1, device=P.device) / (2 * M + 1)
    Q = torch.nn.functional.conv1d(P, kernel, padding=M, groups=L)
    return Q


def asymmetric_filter(Q_in: torch.Tensor, la: float = 0.999, lb: float = 0.5) -> torch.Tensor:
    """Vectorised asymmetric low‑pass along time axis.
       Q_in: [L,T]  returns [L,T]
    """
    if Q_in.dim() == 2:
        Q_in = Q_in.unsqueeze(0)  # [1, L, T]
    elif Q_in.dim() == 3:
        pass
    else:
        raise ValueError(f"asymmetric_filter expects 2D or 3D tensor, got {Q_in.shape}")

    B, L, T = Q_in.shape
    Q_out = torch.empty_like(Q_in)
    # init
    Q_out[:, :, 0] = 0.9 * Q_in[:, :, 0]
    for t in range(1, T):
        rise = Q_in[:, :, t] >= Q_out[:, :, t - 1]
        Q_out[:, :, t] = torch.where(
            rise,
            la * Q_out[:, :, t - 1] + (1 - la) * Q_in[:, :, t],
            lb * Q_out[:, :, t - 1] + (1 - lb) * Q_in[:, :, t],
        )
    return Q_out


def half_wave_rectifier(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp_min_(x, 0.0)


def temporal_masking(Q0: torch.Tensor, lt: float = 0.85, mu_t: float = 0.2) -> torch.Tensor:
    """Implements paper Eq.(11)–(12). Q0 shape [B, L,T]"""
    if Q0.dim() == 2:
        Q0 = Q0.unsqueeze(0)  # [1, L, T]
    elif Q0.dim() != 3:
        raise ValueError(f"asymmetric_filter expects 2D or 3D tensor, got {Q0.shape}")

    B, L, T = Q0.shape
    Qp = torch.empty_like(Q0)
    Qtm = torch.empty_like(Q0)
    Qp[:, :, 0] = Q0[:, :, 0]
    Qtm[:, :, 0] = Q0[:, :, 0]

    for t in range(1, T):
        Qp[:, :, t] = torch.maximum(lt * Qp[:, :, t - 1], Q0[:, :, t])
        Qtm[:, :, t] = torch.where(
            Q0[:, :, t] >= lt * Qp[:, :, t - 1],
            Q0[:, :, t],
            mu_t * Qp[:, :, t - 1],
        )
    Qtm[:, :, 0] = Q0[:, :, 0]  # first frame untouched
    return Qtm


def excitation_switch(Q_tm: torch.Tensor, Q_f: torch.Tensor, Q_med: torch.Tensor, Q_le: torch.Tensor,
                      c: float = 2.0) -> torch.Tensor:
    Q_1 = torch.maximum(Q_tm, Q_f)
    mask_exc = Q_med >= c * Q_le  # [B, L, T]
    return torch.where(mask_exc, Q_1, Q_f)  # [B, L, T]


def spectral_smoothing(R: torch.Tensor, Q: torch.Tensor, N: int = 4) -> torch.Tensor:
    if R.dim() == 2:
        R = R.unsqueeze(0)
        Q = Q.unsqueeze(0)
    elif R.dim() != 3:
        raise ValueError(f"asymmetric_filter expects 2D or 3D tensor, got {R.shape}")

    B, L, T = R.shape
    S = torch.empty_like(R)
    R_div_Q = R / (Q + 1e-12)
    for b in range(B):
        for l in range(L):
            l1, l2 = max(l - N, 0), min(l + N, L - 1)
            S[b, l] = R_div_Q[b, l1:l2 + 1].mean(dim=0)
    return S


def mean_power_normalization(Tm: torch.Tensor, lam: float = 0.999) -> torch.Tensor:
    if Tm.dim() == 2:
        Tm = Tm.unsqueeze(0)
    elif Tm.dim() != (3):
        raise ValueError(f"Expected 2D/3D tensor, got {Tm.shape}")

    B, L, T = Tm.shape
    mu = torch.empty(B, T, device=Tm.device, dtype=Tm.dtype)
    mu[0] = Tm[:, :, 0].mean(dim=1)

    for t in range(1, T):
        mu[:, t] = lam * mu[:, t - 1] + (1 - lam) * Tm[:, :, t].mean(dim=1)

    eps = 1e-12
    T_norm = Tm / (mu.unsqueeze(1) + eps)
    return T_norm


def power_nonlinearity(U: torch.Tensor, exp: float = 1 / 15) -> torch.Tensor:
    return torch.pow(U, exp)


def dct_feats(n_ceps: int, V: torch.Tensor) -> torch.Tensor:
    """Apply DCT (type‑II) across channel axis → cepstra [n_ceps, T]"""
    if V.dim() == 2:
        L, T = V.shape
        k = torch.arange(L, device=V.device)
        basis = torch.cos(np.pi / L * (k + 0.5).unsqueeze(0) * torch.arange(n_ceps, device=V.device).unsqueeze(1))
        return basis @ V
    elif V.dim() == 3:
        B, L, T = V.shape
        k = torch.arange(L, device=V.device)
        basis = torch.cos(np.pi / L * (k + 0.5).unsqueeze(0) * torch.arange(n_ceps, device=V.device).unsqueeze(1))
        # return basis @ V  # [n_ceps,T]
        return torch.einsum('kl,blt->bkt', basis, V)


def pncc(
        wavs: torch.Tensor,
        sr: int,
        n_fft: int = 1024,
        win_length: int = 400,
        hop_length: int = 160,
        n_ch: int = 40,
        f_min: int = 200,
        f_max: int = 8000,
        n_ceps: int = 13,
) -> torch.Tensor:
    """Compute PNCC with fully configurable STFT and filterbank parameters.

    Args:
      wav:      [T] mono waveform.
      sr:       sample rate (Hz).
      n_fft:    FFT size.
      win_length: STFT window length (seconds).
      hop_length:  STFT hop length (seconds).
      n_ch:     number of gammatone channels.
      f_min:    lowest center-frequency (Hz).
      f_max:    highest center-frequency (Hz).
      M, la, lb, lt, mu_t: PNCC-specific hyperparameters.
      n_ceps:   number of DCT cepstra to return.
      power:    exponent for power-law nonlinearity.
    Returns:
      C: [n_ceps, T_frames] PNCC feature matrix.
    """
    if sr != sr:
        raise ValueError(f"sample‑rate {sr} unsupported; expected {sr}")
    if wavs.dim() == 1:
        wavs = wavs.unsqueeze(0)

    # 1. Pre‑emphasis
    wavs = pre_emphasis(wavs)  # supports [B, T] or [T]

    # 2. STFT power
    P_spec = stft_power(
        wavs,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )  # [F, T]

    # 3. Gammatone filterbank power P[m,l]
    H = build_gammatone_bank(
        sr=sr,
        n_fft=n_fft,
        n_ch=n_ch,
        f_min=f_min,
        f_max=f_max,
    ).to(P_spec.device, P_spec.dtype)
    P_ml = apply_filterbank(P_spec, H)  # [L,T]

    # 4. Medium‑time average Q̃
    Q_med = medium_time_avg(P_ml)

    # 5. ANS – lower envelope Q̃_le
    Q_le = asymmetric_filter(Q_med)

    # 6. Half‑wave rectification Q0
    Q0 = half_wave_rectifier(Q_med - Q_le)

    # 7. Floor envelope Q̃_f (asymmetric filter of Q0)
    Q_f = asymmetric_filter(Q0)

    # 8. Temporal masking
    Q_tm = temporal_masking(Q_f)

    # 9. Compose excitation / non‑excitation → R̃ (Eq.10)
    R = excitation_switch(Q_tm, Q_f, Q_med, Q_le)

    # 10. Spectral weight smoothing S̃
    S = spectral_smoothing(R, Q_med)

    # 11. Apply weights to short‑time power → T[m,l]
    T_ml = P_ml * S

    # 12. Mean‑power normalization (online, no look‑ahead)
    U_ml = mean_power_normalization(T_ml)

    # 13. Power‑law nonlinearity
    V_ml = power_nonlinearity(U_ml)

    # 14. DCT → cepstra
    C = dct_feats(n_ceps, V_ml)  # [B, 13,T]

    return C  # [B, 13, T]
