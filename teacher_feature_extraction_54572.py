import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
np.random.seed(3)

def create_student_original_teacher_side_by_side(
    student_embeddings,
    teacher_embeddings,
    orig_img,
    epoch,
    output_heatmaps,
    image_name,
    is_overfit_image=False,
    save_embeddings=False,
):
    """
    Visualizza teacher e student embeddings con colori coerenti.
    Se is_overfit_image=True → calcola la PCA dai teacher embeddings e la salva/carica localmente.
    Se False → calcola la PCA dinamicamente dai teacher embeddings (senza salvataggio/caricamento su disco).
    Salva anche gli embeddings se save_embeddings=True.
    """

    # --- Step 1: gestisci caricamento/salvataggio base PCA ---
    if is_overfit_image:
        print("[INFO] Computing PCA basis from teacher embeddings (no cache)")
        feats = teacher_embeddings.clone().detach().to("cpu")
        if feats.dim() == 4:
            feats = feats[0]  # [C, H, W]
        feats = feats.permute(1, 2, 0).contiguous().reshape(-1, feats.shape[0])  # [H*W, C]
        U, S, V = torch.pca_lowrank(feats, q=3, center=True)
        basis = {"V": V[:, :3], "mean": feats.mean(0)}
    else:
        feats = teacher_embeddings.clone().detach().to("cpu")
        if feats.dim() == 4:
            feats = feats[0]  # [C, H, W]
        feats = feats.permute(1, 2, 0).contiguous().reshape(-1, feats.shape[0])  # [H*W, C]
        U, S, V = torch.pca_lowrank(feats, q=3, center=True)
        basis = {"V": V[:, :3], "mean": feats.mean(0)}

    # --- Step 2: funzione helper per proiettare embeddings con la base caricata ---
    def project_with_basis(embeddings, basis):
        feats = embeddings.clone().detach().to("cpu")
        if feats.dim() == 4:
            feats = feats[0]
        feats = feats.permute(1, 2, 0).reshape(-1, feats.shape[0])  # [H*W, C]
        feats_centered = feats - basis["mean"]
        proj = feats_centered @ basis["V"]  # [H*W, 3]
        proj -= proj.min(0, keepdim=True)[0]
        proj /= proj.max(0, keepdim=True)[0].clamp(min=1e-6)
        H, W = embeddings.shape[-2:]
        rgb = proj.reshape(H, W, 3)
        pil_img = Image.fromarray((rgb.cpu().numpy() * 255).astype("uint8"))
        return pil_img

    # --- Step 3: proietta teacher e student sulla stessa base ---
    pil_img_teacher = project_with_basis(teacher_embeddings, basis)
    pil_img_student = project_with_basis(student_embeddings, basis)

    # --- Step 4: crea immagine combinata ---
    orig_img = orig_img.convert("RGB")
    target_size = orig_img.size
    pil_img_student = pil_img_student.resize(target_size, Image.BILINEAR)
    pil_img_teacher = pil_img_teacher.resize(target_size, Image.BILINEAR)
    w, h = target_size
    combined_img = Image.new("RGB", (w * 3, h))
    combined_img.paste(pil_img_student, (0, 0))
    combined_img.paste(orig_img, (w, 0))
    combined_img.paste(pil_img_teacher, (w * 2, 0))

    # Etichette
    draw = ImageDraw.Draw(combined_img)
    font = ImageFont.load_default(size=32)
    label_height = 40
    draw.rectangle([(0, 0), (w, label_height)], fill=(0, 0, 0, 128))
    draw.rectangle([(w, 0), (w * 2, label_height)], fill=(0, 0, 0, 128))
    draw.rectangle([(w * 2, 0), (w * 3, label_height)], fill=(0, 0, 0, 128))
    draw.text((10, 5), "STUDENT EMBEDDINGS", fill=(255, 255, 255), font=font)
    draw.text((w + 10, 5), "ORIGINAL IMAGE", fill=(255, 255, 255), font=font)
    draw.text((w * 2 + 10, 5), "TEACHER EMBEDDINGS", fill=(255, 255, 255), font=font)

    # --- Step 5: salva il risultato ---
    # include image base name together with epoch to make filenames unique
    combined_path = os.path.join(output_heatmaps, f"{image_name}.png")
    combined_img.save(combined_path)

    # --- Step 6: salva gli embeddings se richiesto ---
    if save_embeddings:
        student_dir = Path(output_heatmaps) / "student"
        teacher_dir = Path(output_heatmaps) / "teacher"
        student_dir.mkdir(parents=True, exist_ok=True)
        teacher_dir.mkdir(parents=True, exist_ok=True)
        torch.save(student_embeddings.detach().cpu(), student_dir / f"{image_name}.pt")
        torch.save(teacher_embeddings.detach().cpu(), teacher_dir / f"{image_name}.pt")

sam2_checkpoint = "/cluster/scratch/niacobone/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

output_dir = "/cluster/scratch/niacobone/distillation/dataset/thank_you/images/train"
COCO2017_PATH = "/cluster/scratch/niacobone/distillation/dataset/thank_you/images/train"

# File specifico da rigenerare
target_file = "Thank_You.jpg"
img_path = os.path.join(COCO2017_PATH, target_file)
save_path = os.path.join(output_dir, "Thank_You.pt")

if not os.path.exists(img_path):
    raise FileNotFoundError(f"Immagine non trovata: {img_path}")

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
sam2.eval()
predictor = SAM2ImagePredictor(sam2)

os.makedirs(output_dir, exist_ok=True)

print(f"[INFO] Caricamento immagine: {img_path}")
pil_img = Image.open(img_path).convert("RGB")
np_img = np.array(pil_img)

print(f"[INFO] Estrazione features...")
predictor.set_image(np_img)
backbone_out = predictor.backbone_out
vision_features = backbone_out["vision_features"]  # (1,C,H,W)

create_student_original_teacher_side_by_side(vision_features, vision_features, pil_img, epoch=0, output_heatmaps=output_dir, image_name="thank_you")

print(f"[INFO] Salvataggio in: {save_path}")
torch.save(vision_features.cpu(), save_path)

print(f"[DONE] Feature salvate: {vision_features.shape}")