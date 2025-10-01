import os
import glob
import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2
# from sam2.utils.transforms import SAM2Transforms
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from sam2.sam2_image_predictor import SAM2ImagePredictor

# def preprocess(pil_img, target=1024):
#     arr = torch.from_numpy(np.array(pil_img.convert("RGB"))).permute(2,0,1).float()/255.
#     arr = torch.nn.functional.interpolate(arr.unsqueeze(0), size=(target,target), mode="bilinear", align_corners=False).squeeze(0)
#     mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
#     std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
#     return (arr - mean)/std

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

# sam2_checkpoint = "/cluster/scratch/niacobone/sam2/checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
# image_path = '/cluster/work/igp_psr/niacobone/examples/photos/small_img/000.jpeg'

# UNOFFICIAL
# sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

# batch = preprocess(Image.open(image_path)).unsqueeze(0).to(device)  # 1x3x1024x1024

# with torch.no_grad():
#     backbone_out = sam2.forward_image(batch)            # forward() in image_encoder
#     vision_features = backbone_out["vision_features"]
#     vision_pos_enc = backbone_out["vision_pos_enc"]
#     backbone_fpn = backbone_out["backbone_fpn"]

#     save_dir = "/cluster/scratch/niacobone/sam2/comparison/unofficial"
#     torch.save(vision_features, f"{save_dir}/vision_features.pt")
#     torch.save(vision_pos_enc, f"{save_dir}/vision_pos_enc.pt")
#     torch.save(backbone_fpn, f"{save_dir}/backbone_fpn.pt")

#     print(f"[DEBUG] Saved vision_features to {save_dir}/vision_features.pt")
#     print(f"[DEBUG] Saved vision_pos_enc to {save_dir}/vision_pos_enc.pt")
#     print(f"[DEBUG] Saved backbone_fpn to {save_dir}/backbone_fpn.pt")

#     # output = {
#     #         "vision_features": src,
#     #         "vision_pos_enc": pos,
#     #         "backbone_fpn": features,
#     #     }

############################################################################################################
# OFFICIAL
# image = Image.open(image_path)
# image = np.array(image.convert("RGB"))

# sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

# mask_generator = SAM2AutomaticMaskGenerator(sam2)

# masks = mask_generator.generate(image)

############################################################################################################
# OFFICIAL - image_predictor_example.ipynb
# sam2_checkpoint = "/cluster/scratch/niacobone/sam2/checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
# image_path = '/cluster/work/igp_psr/niacobone/examples/photos/small_img/000.jpeg'

# save_dir = "/cluster/work/igp_psr/niacobone/sam2/teacher_features"

# image = Image.open(image_path)
# image = np.array(image.convert("RGB"))

# sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

# predictor = SAM2ImagePredictor(sam2)

# predictor.set_image(image)  # this calls backbone_out = self.model.forward_image(input_image)

# backbone_out = predictor.backbone_out

# vision_features = backbone_out["vision_features"]

# torch.save(vision_features, f"{save_dir}/vision_features.pt")

# print(f"[DEBUG] Saved vision_features to {save_dir}/vision_features.pt")

############################################################################################################
# OFFICIAL - image_predictor_example.ipynb - MULTI-FRAME BATCH
sam2_checkpoint = "/cluster/scratch/niacobone/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Percorso singola immagine
image_path = "/cluster/work/igp_psr/niacobone/examples/photos/small_img/000.jpeg"
# Directory con più frame (imposta se multiple_frames=True)
frames_dir = "/cluster/work/igp_psr/niacobone/examples/photos/car_drift"  # esempio
frames_glob = "*.png"  # pattern dei frame

save_dir = "/cluster/work/igp_psr/niacobone/sam2/teacher_features"
os.makedirs(save_dir, exist_ok=True)

multiple_frames = True

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
sam2.eval()
predictor = SAM2ImagePredictor(sam2)

if not multiple_frames:
    # ----- SINGOLA IMMAGINE -----
    pil_img = Image.open(image_path).convert("RGB")
    np_img = np.array(pil_img)
    predictor.set_image(np_img)  # chiamata ufficiale
    backbone_out = predictor.backbone_out
    vision_features = backbone_out["vision_features"]          # (1,C,H,W)
    # backbone_fpn = backbone_out["backbone_fpn"]                # lista livelli
    # vision_pos_enc = backbone_out["vision_pos_enc"]            # lista pos

    torch.save(vision_features.cpu(), f"{save_dir}/vision_features.pt")
    # torch.save(backbone_fpn,         f"{save_dir}/backbone_fpn.pt")
    # torch.save(vision_pos_enc,       f"{save_dir}/vision_pos_enc.pt")
    print(f"[INFO] Salvato feature singola immagine in {save_dir}")

else:
    # ----- MULTI-FRAME BATCH -----
    # 1. Raccogli lista frame
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, frames_glob)))
    if len(frame_paths) == 0:
        raise RuntimeError(f"Nessun frame trovato in {frames_dir} con pattern {frames_glob}")

    frames = [np.array(Image.open(p).convert("RGB")) for p in frame_paths]
    predictor.set_image_batch(frames)  # crea batch e fa forward

    backbone_out = predictor.backbone_out

    vision_features = backbone_out["vision_features"]      # (B,C,H,W)
    # backbone_fpn = backbone_out["backbone_fpn"]            # lista di L tensori (B,C,h_i,w_i)
    # vision_pos_enc = backbone_out["vision_pos_enc"]        # lista pos (L)

    # Salvataggio batch intero
    torch.save(vision_features.cpu(), f"{save_dir}/vision_features_batch.pt")
    # torch.save(backbone_fpn,         f"{save_dir}/backbone_fpn_batch.pt")
    # torch.save(vision_pos_enc,       f"{save_dir}/vision_pos_enc_batch.pt")

    # (Opzionale) salvataggio per-frame
    # per_frame_dir = os.path.join(save_dir, "per_frame")
    # os.makedirs(per_frame_dir, exist_ok=True)
    # for i in range(vision_features.size(0)):
    #     torch.save(vision_features[i].cpu(),
    #                f"{per_frame_dir}/vision_features_{i:04d}.pt")
    print(f"[INFO] Salvate feature batch ({vision_features.size(0)} frame) in {save_dir}")

print("[DONE]")