import torch
import torch.nn as nn
import torch.nn.functional as F

from .scan import g, log_g, naive_sequential_scan, vanilla_scan
from .min_gru import _scan_log_with_h0


class minLSTM(nn.Module):
    def __init__(self, dim, hidden_dim=None, proj_out=False):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.to_f = nn.Linear(dim, hidden_dim, bias=False)
        self.to_i = nn.Linear(dim, hidden_dim, bias=False)
        self.to_h = nn.Linear(dim, hidden_dim, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False) if proj_out else nn.Identity()

    def _pre(self, x):
        return self.to_f(x), self.to_i(x), self.to_h(x)

    def forward_parallel_log(self, x, h0=None):
        pre_f, pre_i, pre_h = self._pre(x)
        log_f = -F.softplus(-pre_f)
        log_i = -F.softplus(-pre_i)
        log_denom = torch.logaddexp(log_f, log_i)
        log_a = log_f - log_denom
        log_b = (log_i - log_denom) + log_g(pre_h)
        h = _scan_log_with_h0(log_a, log_b, h0)
        return self.to_out(h)

    def _ab(self, x):
        pre_f, pre_i, pre_h = self._pre(x)
        f = pre_f.sigmoid()
        i = pre_i.sigmoid()
        denom = f + i
        return f / denom, (i / denom) * g(pre_h)

    def forward_parallel_vanilla(self, x, h0=None):
        a, b = self._ab(x)
        return self.to_out(vanilla_scan(a, b, h0))

    def forward_sequential(self, x, h0=None):
        a, b = self._ab(x)
        return self.to_out(naive_sequential_scan(a, b, h0))

    def forward(self, x, h0=None, mode="parallel_log"):
        if mode == "parallel_log":
            return self.forward_parallel_log(x, h0)
        if mode == "parallel_vanilla":
            return self.forward_parallel_vanilla(x, h0)
        if mode == "sequential":
            return self.forward_sequential(x, h0)
        raise ValueError(f"unknown mode: {mode}")
