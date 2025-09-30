"""Script di confronto feature SAM2

Confronta le feature ottenute con preprocessing manuale (funzione preprocess)
e quelle ottenute tramite la pipeline ufficiale (SAM2ImagePredictor + SAM2Transforms).

Esegue:
 1. Caricamento modello SAM2
 2. Selezione immagine (da --image oppure prima valida in --image_dir)
 3. Estrazione feature manuale via sam2.forward_image
 4. Estrazione feature ufficiale via predictor.set_image
 5. Confronto vision_features, livelli FPN, positional encodings
"""

import argparse
import glob
import hashlib
import os
from typing import List

import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def select_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    print(f"[INFO] Using device: {dev}")
    if dev.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    return dev


_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess(pil_img: Image.Image, target: int = 1024) -> torch.Tensor:
    arr = torch.from_numpy(np.array(pil_img.convert("RGB"))).permute(2, 0, 1).float() / 255.0
    if arr.shape[1] != target or arr.shape[2] != target:
        arr = torch.nn.functional.interpolate(
            arr.unsqueeze(0), size=(target, target), mode="bilinear", align_corners=False
        ).squeeze(0)
    return (arr - _MEAN) / _STD


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


def compare_features(a: torch.Tensor, b: torch.Tensor, name: str = "vision_features", atol=1e-5, rtol=1e-4):
    a_f = a.detach().to(dtype=torch.float32)
    b_f = b.detach().to(dtype=torch.float32)
    diff = a_f - b_f
    l2 = torch.linalg.norm(diff).item()
    base = torch.linalg.norm(a_f).item()
    max_abs = diff.abs().max().item()
    rel = l2 / (base + 1e-12)
    close = torch.allclose(a_f, b_f, atol=atol, rtol=rtol)
    over = (diff.abs() > atol + rtol * a_f.abs()).sum().item()
    print(f"\n=== Confronto {name} ===")
    print(stats_str("A(manuale)", a))
    print(stats_str("B(ufficiale)", b))
    print(
        f"L2 diff={l2:.6f}  rel_L2={rel:.6e}  max_abs={max_abs:.6e}  "
        f"allclose={close}  n_elem_over_tol={over}"
    )


def find_image_paths(image: str, image_dir: str, exts: List[str]) -> List[str]:
    if image:
        return [image]
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Directory non trovata: {image_dir}")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(image_dir, f"*{e}")))
    if not files:
        raise RuntimeError("Nessuna immagine trovata nella cartella specificata")
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser(description="Confronto feature SAM2 manual vs ufficiale")
    parser.add_argument("--image", type=str, default=None, help="Percorso immagine singola")
    parser.add_argument(
        "--image_dir", type=str, default="/cluster/work/igp_psr/niacobone/examples/photos/pianta",
        help="Directory immagini (usata se --image non è fornito)",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="/cluster/scratch/niacobone/sam2/checkpoints/sam2.1_hiera_large.pt",
        help="Percorso checkpoint SAM2",
    )
    parser.add_argument(
        "--config", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml", help="Config modello"
    )
    parser.add_argument("--no_pos", action="store_true", help="Non confrontare i positional encodings")
    args = parser.parse_args()

    device = select_device()

    sam2 = build_sam2(
        args.config, args.checkpoint, device=device, apply_postprocessing=False
    )
    sam2.eval()
    image_paths = find_image_paths(args.image, args.image_dir, [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"])
    test_path = image_paths[0]
    print(f"[INFO] Usando immagine: {test_path}")

    pil_img = Image.open(test_path).convert("RGB")
    target = sam2.image_size if hasattr(sam2, "image_size") else 1024

    # Pipeline manuale
    manual_tensor = preprocess(pil_img, target=target).unsqueeze(0).to(device)
    with torch.no_grad():
        out_manual = sam2.forward_image(manual_tensor)
    vf_manual = out_manual["vision_features"]
    fpn_manual = out_manual["backbone_fpn"]
    pos_manual = out_manual["vision_pos_enc"]

    # Pipeline ufficiale
    predictor = SAM2ImagePredictor(sam2)
    predictor.set_image(np.array(pil_img))
    emb = predictor._image_embeddings
    vf_official = emb["vision_features"]
    fpn_official = emb["backbone_fpn"]
    pos_official = emb["vision_pos_enc"]

    # Confronti
    compare_features(vf_manual, vf_official, "vision_features")
    for i, (m_lvl, o_lvl) in enumerate(zip(fpn_manual, fpn_official)):
        compare_features(m_lvl, o_lvl, name=f"fpn_level_{i}")
    if not args.no_pos:
        for i, (pm, po) in enumerate(zip(pos_manual, pos_official)):
            compare_features(pm, po, name=f"pos_enc_level_{i}")

    print("\n[FINITO] Confronto completato.")


if __name__ == "__main__":
    main()