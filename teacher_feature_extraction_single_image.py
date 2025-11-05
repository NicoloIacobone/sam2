import os
import numpy as np
import torch
from PIL import Image

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

sam2_checkpoint = "/cluster/scratch/niacobone/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

output_dir = "/cluster/work/igp_psr/niacobone/distillation/sam2/test_multi_view_single_inference"
INPUT_PATH = "/cluster/work/igp_psr/niacobone/distillation/sam2/test_multi_view_single_inference"

# File specifico da rigenerare
target_file = "00000.jpg"
img_path = os.path.join(INPUT_PATH, target_file)
save_path = os.path.join(output_dir, "00000.pt")

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

print(f"[INFO] Salvataggio in: {save_path}")
torch.save(vision_features.cpu(), save_path)

print(f"[DONE] Feature salvate: {vision_features.shape}")