"""Confronto feature SAM2 tra due pipeline già salvate.

Questo script NON esegue inferenza. Si aspetta due cartelle:
  official/    contenente: vision_features.pt, backbone_fpn.pt, vision_pos_enc.pt
  unofficial/  contenente: vision_features.pt, backbone_fpn.pt, vision_pos_enc.pt

Confronta:
  - vision_features (tensor singolo)
  - backbone_fpn (lista di livelli)
  - vision_pos_enc (lista di livelli, opzionale con --no_pos)

Metriche stampate per ogni confronto:
  shape, dtype, min, max, mean, std, sum, hash md5, L2 diff, max abs diff,
  L2 relativo, allclose, numero elementi fuori tolleranza.
"""

import argparse
import os
import hashlib
import torch
from typing import Sequence

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
    print(stats_str("A(unofficial)", a))
    print(stats_str("B(official)", b))
    print(
        f"L2 diff={l2:.6f}  rel_L2={rel:.6e}  max_abs={max_abs:.6e}  "
        f"allclose={close}  n_elem_over_tol={over}"
    )

def load_tensor(path: str, descr: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File mancante ({descr}): {path}")
    obj = torch.load(path, map_location="cpu")
    return obj

def ensure_sequence(obj, name: str) -> Sequence[torch.Tensor]:
    if isinstance(obj, (list, tuple)):
        return obj
    raise TypeError(f"Il file {name} deve contenere una lista/tupla di tensori, trovato: {type(obj)}")

def main():
    parser = argparse.ArgumentParser(description="Confronta feature SAM2 tra due cartelle (official/unofficial)")
    parser.add_argument("--official_dir", type=str, required=False, default="/cluster/scratch/niacobone/sam2/comparison/official", help="Cartella con i file ufficiali")
    parser.add_argument("--unofficial_dir", type=str, required=False, default="/cluster/scratch/niacobone/sam2/comparison/unofficial", help="Cartella con i file unofficial")
    parser.add_argument("--no_pos", action="store_true", help="Non confrontare i positional encodings")
    parser.add_argument("--atol", type=float, default=1e-5, help="Tolleranza assoluta allclose")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Tolleranza relativa allclose")
    args = parser.parse_args()

    print("[INFO] Cartelle input:")
    print("  official:   ", args.official_dir)
    print("  unofficial: ", args.unofficial_dir)

    files = ["vision_features.pt", "backbone_fpn.pt", "vision_pos_enc.pt"]
    paths_off = {f: os.path.join(args.official_dir, f) for f in files}
    paths_unoff = {f: os.path.join(args.unofficial_dir, f) for f in files}

    # Caricamento
    vf_off = load_tensor(paths_off["vision_features.pt"], "vision_features official")
    vf_unoff = load_tensor(paths_unoff["vision_features.pt"], "vision_features unofficial")
    fpn_off = load_tensor(paths_off["backbone_fpn.pt"], "backbone_fpn official")
    fpn_unoff = load_tensor(paths_unoff["backbone_fpn.pt"], "backbone_fpn unofficial")
    pos_off = load_tensor(paths_off["vision_pos_enc.pt"], "vision_pos_enc official")
    pos_unoff = load_tensor(paths_unoff["vision_pos_enc.pt"], "vision_pos_enc unofficial")

    # Vision features
    compare_features(vf_unoff, vf_off, "vision_features", args.atol, args.rtol)

    # Backbone FPN
    fpn_off_seq = ensure_sequence(fpn_off, "backbone_fpn official")
    fpn_unoff_seq = ensure_sequence(fpn_unoff, "backbone_fpn unofficial")
    print("\n[INFO] Confronto backbone_fpn (levels)")
    if len(fpn_off_seq) != len(fpn_unoff_seq):
        print(f"[ERRORE] Lunghezza diversa FPN: off={len(fpn_off_seq)} unoff={len(fpn_unoff_seq)}")
    for i, (tu, to) in enumerate(zip(fpn_unoff_seq, fpn_off_seq)):
        compare_features(tu, to, f"fpn_level_{i}", args.atol, args.rtol)

    # Positional encodings
    if not args.no_pos:
        pos_off_seq = ensure_sequence(pos_off, "vision_pos_enc official")
        pos_unoff_seq = ensure_sequence(pos_unoff, "vision_pos_enc unofficial")
        print("\n[INFO] Confronto vision_pos_enc (levels)")
        if len(pos_off_seq) != len(pos_unoff_seq):
            print(f"[ERRORE] Lunghezza diversa pos_enc: off={len(pos_off_seq)} unoff={len(pos_unoff_seq)}")
        for i, (tu, to) in enumerate(zip(pos_unoff_seq, pos_off_seq)):
            compare_features(tu, to, f"pos_enc_level_{i}", args.atol, args.rtol)

    print("\n[FINITO] Confronto completato.")

if __name__ == "__main__":
    main()