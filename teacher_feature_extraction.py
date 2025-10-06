import os
import glob
import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def is_image_file(name: str) -> bool:
    name_low = name.lower()
    return name_low.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
np.random.seed(3)

# OFFICIAL - image_predictor_example.ipynb - MULTI-FRAME BATCH
sam2_checkpoint = "/cluster/scratch/niacobone/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Percorso singola immagine
image_path = "/cluster/work/igp_psr/niacobone/examples/photos/small_img/000.jpeg"

dir_names = ["yokohama", "tenda_ufficio", "sedia_ufficio", "pianta", "car_drift"]
base_path = "/cluster/work/igp_psr/niacobone"
output_dir = "/cluster/scratch/niacobone/distillation/sam2/coco2017"
COCO2017_PATH = "/cluster/work/igp_psr/niacobone/coco2017"
# frames_glob deve essere una lista, appaiata con dir_names
frames_globs = ["*.jpg", "*.png", "*.png", "*.png", "*.png"]

# Directory con più frame (imposta se multiple_frames=True)
multiple_frames = False

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
sam2.eval()
predictor = SAM2ImagePredictor(sam2)

os.makedirs(output_dir, exist_ok=True)
image_paths = [os.path.join(COCO2017_PATH, f) for f in os.listdir(COCO2017_PATH) if is_image_file(f)]
image_paths.sort()

if not image_paths:
    print(f"[WARNING] Nessuna immagine trovata in {COCO2017_PATH}")
    raise Exception("Nessuna immagine trovata")
print(f"[INFO] Trovate {len(image_paths)} immagini in {COCO2017_PATH}. Salvataggio in {output_dir}")

for idx, img_path in enumerate(image_paths[:5]):
    try:
        pil_img = Image.open(img_path).convert("RGB")
        np_img = np.array(pil_img)
        predictor.set_image(np_img)
        backbone_out = predictor.backbone_out
        vision_features = backbone_out["vision_features"]  # (1,C,H,W)
        stem = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(output_dir, f"{stem}.pt")
        torch.save(vision_features.cpu(), save_path)
        print(f"  -> {idx+1}/5 salvate (ultima: {os.path.basename(save_path)})")
        # if (idx + 1) % 50 == 0 or idx == len(image_paths) - 1:
        #     print(f"  -> {idx+1}/{len(image_paths)} salvate (ultima: {os.path.basename(save_path)})")
    except Exception as e:
        print(f"[ERRORE] Immagine {img_path} saltata: {e}")

# for dir_name, frames_glob in zip(dir_names, frames_globs):
#     input_path = os.path.join(base_path, "examples/photos", dir_name)
#     output_path = os.path.join(base_path, "distillation/sam2", dir_name)
#     os.makedirs(output_path, exist_ok=True)

#     if not multiple_frames:
#         # ----- SINGOLA IMMAGINE -----
#         pil_img = Image.open(image_path).convert("RGB")
#         np_img = np.array(pil_img)
#         predictor.set_image(np_img)  # chiamata ufficiale
#         backbone_out = predictor.backbone_out
#         vision_features = backbone_out["vision_features"]          # (1,C,H,W)
#         torch.save(vision_features.cpu(), f"{output_path}/vision_features.pt")
#         print(f"[INFO] Salvato feature singola immagine in {output_path}")

#     else:
#         # ----- MULTI-FRAME BATCH -----
#         # 1. Raccogli lista frame
#         frame_paths = sorted(glob.glob(os.path.join(input_path, frames_glob)))
#         if len(frame_paths) == 0:
#             print(f"[WARNING] Nessun frame trovato in {input_path} con pattern {frames_glob}")
#             continue

#         frames = [np.array(Image.open(p).convert("RGB")) for p in frame_paths]
#         predictor.set_image_batch(frames)  # crea batch e fa forward

#         backbone_out = predictor.backbone_out

#         vision_features = backbone_out["vision_features"]        # (B,C,H,W)

#         # Salvataggio batch intero
#         torch.save(vision_features.cpu(), f"{output_path}/teacher_embeddings.pt")

#         print(f"[INFO] Salvate feature batch ({vision_features.size(0)} frame) in {output_path}")