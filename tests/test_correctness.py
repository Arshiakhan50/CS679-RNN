import itertools
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import minGRU, minLSTM  # noqa: E402

MODELS = [minGRU, minLSTM]
SHAPES = [(2, 16, 8), (3, 64, 32), (1, 128, 16)]


def _model(cls, dim, seed=0):
    torch.manual_seed(seed)
    return cls(dim).double()


def test_parallel_log_matches_sequential():
    for cls, (B, T, D) in itertools.product(MODELS, SHAPES):
        M = _model(cls, D)
        x = torch.randn(B, T, D, dtype=torch.float64)
        err = (M(x, mode="sequential") - M(x, mode="parallel_log")).abs().max().item()
        assert err < 1e-8, f"{cls.__name__} {(B,T,D)} log-space err={err:.2e}"


def test_vanilla_matches_sequential_short():
    for cls in MODELS:
        B, T, D = 2, 16, 8
        M = _model(cls, D)
        x = torch.randn(B, T, D, dtype=torch.float64)
        err = (M(x, mode="sequential") - M(x, mode="parallel_vanilla")).abs().max().item()
        assert err < 1e-8, f"{cls.__name__} vanilla err={err:.2e}"


def test_gradients_finite_and_nonzero():
    for cls in MODELS:
        B, T, D = 2, 32, 16
        M = _model(cls, D)
        x = torch.randn(B, T, D, dtype=torch.float64)
        M(x, mode="parallel_log").sum().backward()
        for n, p in M.named_parameters():
            assert p.grad is not None, f"{cls.__name__}.{n} has no grad"
            assert torch.isfinite(p.grad).all(), f"{cls.__name__}.{n} non-finite grad"
            assert p.grad.abs().sum() > 0, f"{cls.__name__}.{n} zero grad"


def test_gradcheck():
    for cls in MODELS:
        B, T, D = 1, 8, 4
        M = _model(cls, D)
        x = torch.randn(B, T, D, dtype=torch.float64, requires_grad=True)
        ok = torch.autograd.gradcheck(
            lambda inp: M(inp, mode="parallel_log"), (x,), eps=1e-6, atol=1e-4
        )
        assert ok, f"{cls.__name__} gradcheck failed"


def test_overfit_tiny():
    # A single minimal layer is low-capacity, so give it width + an output
    # projection; the point is only to show the training path reduces loss.
    for cls in MODELS:
        B, T, Din = 1, 10, 8
        torch.manual_seed(0)
        M = cls(Din, hidden_dim=64, proj_out=True)
        x = torch.randn(B, T, Din)
        y = torch.randn(B, T, Din)
        opt = torch.optim.Adam(M.parameters(), lr=3e-3)
        loss0 = None
        for step in range(1500):
            opt.zero_grad()
            loss = ((M(x, mode="parallel_log") - y) ** 2).mean()
            loss.backward()
            opt.step()
            if step == 0:
                loss0 = loss.item()
        assert loss.item() < 0.05 * loss0, (
            f"{cls.__name__} failed to overfit: {loss0:.4f} -> {loss.item():.4f}"
        )


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
