import torch
import torch.nn.functional as F


def g(x):
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))


def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))


def heinsen_associative_scan_log(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(dim=1)
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()


def naive_sequential_scan(a, b, h0=None):
    B, T, D = a.shape
    h_prev = torch.zeros(B, D, dtype=a.dtype, device=a.device) if h0 is None else h0
    outs = []
    for t in range(T):
        h_prev = a[:, t] * h_prev + b[:, t]
        outs.append(h_prev)
    return torch.stack(outs, dim=1)


def vanilla_scan(a, b, h0=None):
    A = torch.cumprod(a, dim=1)
    inner = torch.cumsum(b / A, dim=1)
    if h0 is not None:
        inner = inner + h0.unsqueeze(1)
    return A * inner
