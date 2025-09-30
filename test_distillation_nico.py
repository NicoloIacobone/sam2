import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
# import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2
# from sam2.utils.transforms import SAM2Transforms
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def preprocess(pil_img, target=1024):
    arr = torch.from_numpy(np.array(pil_img.convert("RGB"))).permute(2,0,1).float()/255.
    arr = torch.nn.functional.interpolate(arr.unsqueeze(0), size=(target,target), mode="bilinear", align_corners=False).squeeze(0)
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    return (arr - mean)/std

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

sam2_checkpoint = "/cluster/scratch/niacobone/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
image_path = '/cluster/work/igp_psr/niacobone/examples/photos/small_img/000.jpeg'

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
image = Image.open(image_path)
image = np.array(image.convert("RGB"))

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(sam2)

masks = mask_generator.generate(image)