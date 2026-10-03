import torch

from src import minGRU, minLSTM


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}  |  device = {dev}")
    if dev == "cuda":
        print(f"gpu    = {torch.cuda.get_device_name(0)}")
    print()

    torch.manual_seed(0)
    B, T, D = 4, 64, 32
    x = torch.randn(B, T, D, device=dev)

    print(f"{'model':8s} {'shape':>16s} {'max|seq-log|':>14s} {'max|seq-vanilla|':>18s}")
    for name, M in [("minGRU", minGRU(D)), ("minLSTM", minLSTM(D))]:
        M = M.to(dev)
        seq = M(x, mode="sequential")
        plog = M(x, mode="parallel_log")
        pvan = M(x, mode="parallel_vanilla")
        e_log = (seq - plog).abs().max().item()
        e_van = (seq - pvan).abs().max().item()
        print(f"{name:8s} {str(tuple(plog.shape)):>16s} {e_log:>14.2e} {e_van:>18.2e}")

    M = minGRU(D).to(dev)
    M(x, mode="parallel_log").sum().backward()
    print(f"\ngrad norm (minGRU to_z): {M.to_z.weight.grad.norm().item():.4f}")
    print("OK")


if __name__ == "__main__":
    main()
