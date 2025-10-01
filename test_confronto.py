import argparse
import os
import hashlib
import torch

def tensor_hash(t: torch.Tensor) -> str:
    b = t.detach().to("cpu", dtype=torch.float32).numpy().tobytes()
    return hashlib.md5(b).hexdigest()

def stats_str(name: str, t: torch.Tensor) -> str:
    t_cpu = t.detach().to("cpu", dtype=torch.float32)
    return (
        f"{name}: shape={tuple(t_cpu.shape)} dtype={t.dtype} "
        f"min={t_cpu.min():.4f} max={t_cpu.max():.4f} "
        f"mean={t_cpu.mean():.4f} std={t_cpu.std():.4f} "
        f"sum={t_cpu.sum():.4f} hash={tensor_hash(t)}"
    )

def compare_features(a: torch.Tensor, b: torch.Tensor, name: str, atol: float, rtol: float):
    a_f = a.detach().to(dtype=torch.float32)
    b_f = b.detach().to(dtype=torch.float32)
    if a_f.shape != b_f.shape:
        print(f"\n=== Confronto {name} ===")
        print("[ERRORE] Shape diverse:", a_f.shape, b_f.shape)
        return
    diff = a_f - b_f
    l2 = torch.linalg.norm(diff).item()
    base = torch.linalg.norm(a_f).item()
    rel = l2 / (base + 1e-12)
    max_abs = diff.abs().max().item()
    close = torch.allclose(a_f, b_f, atol=atol, rtol=rtol)
    over = (diff.abs() > atol + rtol * a_f.abs()).sum().item()
    print(f"\n=== Confronto {name} ===")
    print(stats_str("A(vision_features.pt)", a))
    print(stats_str("B(vision_features_0.pt)", b))
    print(
        f"L2 diff={l2:.6f}  rel_L2={rel:.6e}  max_abs={max_abs:.6e}  "
        f"allclose={close}  n_elem_over_tol={over}"
    )

def load_tensor(path: str, descr: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File mancante ({descr}): {path}")
    obj = torch.load(path, map_location="cpu")
    return obj

def main():
    parser = argparse.ArgumentParser(description="Confronta solo vision_features tra due file")
    parser.add_argument("--file", type=str, default="/cluster/work/igp_psr/niacobone/sam2/teacher_features/", help="File vision_features")
    parser.add_argument("--atol", type=float, default=1e-5, help="Tolleranza assoluta allclose")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Tolleranza relativa allclose")
    args = parser.parse_args()

    print("[INFO] File input:")
    print("  file: ", args.file)


    vf_a = load_tensor(args.file, "vision_features.pt")
    vf_b = load_tensor(args.file, "vision_features_0.pt")

    compare_features(vf_a, vf_b, "vision_features", args.atol, args.rtol)

    print("\n[FINITO] Confronto completato.")

if __name__ == "__main__":
    main()
