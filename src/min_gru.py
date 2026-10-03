import torch
import torch.nn as nn
import torch.nn.functional as F

from .scan import (
    g,
    log_g,
    heinsen_associative_scan_log,
    naive_sequential_scan,
    vanilla_scan,
)


def _scan_log_with_h0(log_a, log_b, h0=None):
    B, T, D = log_a.shape
    log_a = F.pad(log_a, (0, 0, 1, 0))
    if h0 is None:
        log_h0 = torch.full((B, 1, D), float("-inf"), dtype=log_b.dtype, device=log_b.device)
    else:
        log_h0 = h0.clamp(min=1e-20).log().unsqueeze(1)
    log_b = torch.cat([log_h0, log_b], dim=1)
    h = heinsen_associative_scan_log(log_a, log_b)
    return h[:, 1:]


class minGRU(nn.Module):
    def __init__(self, dim, hidden_dim=None, proj_out=False):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.to_z = nn.Linear(dim, hidden_dim, bias=False)
        self.to_h = nn.Linear(dim, hidden_dim, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False) if proj_out else nn.Identity()

    def _pre(self, x):
        return self.to_z(x), self.to_h(x)

    def forward_parallel_log(self, x, h0=None):
        pre_z, pre_h = self._pre(x)
        log_a = -F.softplus(pre_z)
        log_z = -F.softplus(-pre_z)
        log_b = log_z + log_g(pre_h)
        h = _scan_log_with_h0(log_a, log_b, h0)
        return self.to_out(h)

    def forward_parallel_vanilla(self, x, h0=None):
        pre_z, pre_h = self._pre(x)
        z = pre_z.sigmoid()
        a = 1.0 - z
        b = z * g(pre_h)
        return self.to_out(vanilla_scan(a, b, h0))

    def forward_sequential(self, x, h0=None):
        pre_z, pre_h = self._pre(x)
        z = pre_z.sigmoid()
        a = 1.0 - z
        b = z * g(pre_h)
        return self.to_out(naive_sequential_scan(a, b, h0))

    def forward(self, x, h0=None, mode="parallel_log"):
        if mode == "parallel_log":
            return self.forward_parallel_log(x, h0)
        if mode == "parallel_vanilla":
            return self.forward_parallel_vanilla(x, h0)
        if mode == "sequential":
            return self.forward_sequential(x, h0)
        raise ValueError(f"unknown mode: {mode}")
