# This script uses the features produced by the additional head of the student model to infer using the teacher decoder

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import glob

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)

np.random.seed(3)

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

dir_name = "box_ufficio"
base_path = "/cluster/work/igp_psr/niacobone"
# frames_glob deve essere una lista, appaiata con dir_names
frames_glob = "*.png"

input_path = os.path.join(base_path, "examples/photos", dir_name)
output_path = os.path.join(base_path, "consistency_test", dir_name)
os.makedirs(output_path, exist_ok=True)
frame_paths = sorted(glob.glob(os.path.join(input_path, frames_glob)))
frames = [np.array(Image.open(p).convert("RGB")) for p in frame_paths]

sam2_checkpoint = "/cluster/scratch/niacobone/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(sam2)

# per questo test considero solo la prima immagine
image = frames[0]
masks = mask_generator.generate(image)

plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
# output_file = os.path.join(output_path, "masks_consistency.png")
output_file = os.path.join(output_path, "masks_original.png")
plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
plt.close()