import numpy as np
import scipy
from scipy import stats
import torch

def bandwidth_to_max_bin(rate, n_fft, bandwidth):
    freqs = np.linspace(0, float(rate) / 2, n_fft // 2 + 1, endpoint=True)

    return np.max(np.where(freqs <= bandwidth)[0]) + 1

def gen_poisson(mu):
    """Adapted from https://github.com/rmalouf/learning/blob/master/zt.py"""
    r = np.random.uniform(low=stats.poisson.pmf(0, mu))
    return stats.poisson.ppf(r, mu)

def snr(pred_signal: torch.Tensor, true_signal: torch.Tensor) -> torch.FloatTensor:
    """
    Calculate the Signal-to-Noise Ratio
    from two signals
    Args:
        pred_signal (torch.Tensor): predicted signal.
        true_signal (torch.Tensor): original signal.
    """
    inter_signal = true_signal - pred_signal

    true_power = (true_signal ** 2).sum()
    inter_power = (inter_signal ** 2).sum()

    snr = 10 * torch.log10(true_power / inter_power)

    return snr