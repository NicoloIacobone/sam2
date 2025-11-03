import os
import numpy as np
import torch
from PIL import Image
import argparse

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def is_image_file(name: str) -> bool:
    name_low = name.lower()
    return name_low.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))

# Argomenti CLI
parser = argparse.ArgumentParser()
parser.add_argument("--split-id", type=int, required=True, help="ID dello split (0-based, es. 0-9 per 10 split)")
parser.add_argument("--num-splits", type=int, default=10, help="Numero totale di split")
args = parser.parse_args()

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

output_dir = "/cluster/scratch/niacobone/distillation/dataset/coco2017/train2017"
COCO2017_PATH = "/cluster/work/igp_psr/data/cocostuff/dataset/images/train2017"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
sam2.eval()
predictor = SAM2ImagePredictor(sam2)

os.makedirs(output_dir, exist_ok=True)
image_paths = [os.path.join(COCO2017_PATH, f) for f in os.listdir(COCO2017_PATH) if is_image_file(f)]
image_paths.sort()

if not image_paths:
    print(f"[WARNING] Nessuna immagine trovata in {COCO2017_PATH}")
    raise Exception("Nessuna immagine trovata")

# Dividi il dataset in split
total_images = len(image_paths)
split_size = (total_images + args.num_splits - 1) // args.num_splits  # ceiling division
start_idx = args.split_id * split_size
end_idx = min(start_idx + split_size, total_images)
my_images = image_paths[start_idx:end_idx]

print(f"[INFO] Split {args.split_id}/{args.num_splits}: processa {len(my_images)} immagini ({start_idx}-{end_idx-1} di {total_images})")
print(f"[INFO] Output in {output_dir}")

processed = 0
skipped = 0

for idx, img_path in enumerate(my_images):
    try:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(output_dir, f"{stem}.pt")
        
        # Skip se già processata
        if os.path.exists(save_path):
            skipped += 1
            if (idx + 1) % 100 == 0:
                print(f"  -> {idx+1}/{len(my_images)} | processate: {processed}, skippate: {skipped}")
            continue
        
        pil_img = Image.open(img_path).convert("RGB")
        np_img = np.array(pil_img)
        predictor.set_image(np_img)
        backbone_out = predictor.backbone_out
        vision_features = backbone_out["vision_features"]  # (1,C,H,W)
        torch.save(vision_features.cpu(), save_path)
        processed += 1
        
        if (idx + 1) % 100 == 0 or idx == len(my_images) - 1:
            print(f"  -> {idx+1}/{len(my_images)} | processate: {processed}, skippate: {skipped} (ultima: {os.path.basename(save_path)})")
    except Exception as e:
        print(f"[ERRORE] Immagine {img_path} saltata: {e}")

print(f"[DONE] Split {args.split_id} completato: {processed} processate, {skipped} skippate")

# import os
# import glob
# import numpy as np
# import torch
# from PIL import Image

# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor

# def is_image_file(name: str) -> bool:
#     name_low = name.lower()
#     return name_low.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))

# # select the device for computation
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
# print(f"using device: {device}")

# if device.type == "cuda":
#     # use bfloat16 for the entire notebook
#     torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
#     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#     if torch.cuda.get_device_properties(0).major >= 8:
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True
# np.random.seed(3)

# # OFFICIAL - image_predictor_example.ipynb - MULTI-FRAME BATCH
# sam2_checkpoint = "/cluster/scratch/niacobone/sam2/checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# # Percorso singola immagine
# # image_path = "/cluster/work/igp_psr/niacobone/examples/photos/small_img/000.jpeg"

# # dir_names = ["yokohama", "tenda_ufficio", "sedia_ufficio", "pianta", "car_drift"]
# # base_path = "/cluster/work/igp_psr/niacobone"
# # frames_globs = ["*.jpg", "*.png", "*.png", "*.png", "*.png"]
# # frames_glob deve essere una lista, appaiata con dir_names
# # Directory con più frame (imposta se multiple_frames=True)
# multiple_frames = False

# # output_dir = "/cluster/scratch/niacobone/distillation/sam2/coco2017"
# # COCO2017_PATH = "/cluster/work/igp_psr/niacobone/coco2017"
# output_dir = "/cluster/scratch/niacobone/distillation/dataset/coco2017/val2017"
# COCO2017_PATH = "/cluster/work/igp_psr/data/cocostuff/dataset/images/val2017"

# sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
# sam2.eval()
# predictor = SAM2ImagePredictor(sam2)

# os.makedirs(output_dir, exist_ok=True)
# image_paths = [os.path.join(COCO2017_PATH, f) for f in os.listdir(COCO2017_PATH) if is_image_file(f)]
# image_paths.sort()

# if not image_paths:
#     print(f"[WARNING] Nessuna immagine trovata in {COCO2017_PATH}")
#     raise Exception("Nessuna immagine trovata")
# print(f"[INFO] Trovate {len(image_paths)} immagini in {COCO2017_PATH}. Salvataggio in {output_dir}")

# for idx, img_path in enumerate(image_paths):
#     try:
#         stem = os.path.splitext(os.path.basename(img_path))[0]
#         save_path = os.path.join(output_dir, f"{stem}.pt")
        
#         # Skip se già processata
#         if os.path.exists(save_path):
#             if (idx + 1) % 500 == 0:
#                 print(f"  -> {idx+1}/{len(image_paths)} (skip: {os.path.basename(save_path)} già esistente)")
#             continue
        
#         pil_img = Image.open(img_path).convert("RGB")
#         np_img = np.array(pil_img)
#         predictor.set_image(np_img)
#         backbone_out = predictor.backbone_out
#         vision_features = backbone_out["vision_features"]  # (1,C,H,W)
#         torch.save(vision_features.cpu(), save_path)
#         # print(f"  -> {idx+1}/50 salvate (ultima: {os.path.basename(save_path)})")
#         if (idx + 1) % 500 == 0 or idx == len(image_paths) - 1:
#             print(f"  -> {idx+1}/{len(image_paths)} salvate (ultima: {os.path.basename(save_path)})")
#     except Exception as e:
#         print(f"[ERRORE] Immagine {img_path} saltata: {e}")

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