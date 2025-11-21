# This script uses the features produced by the additional head of the student model to infer using the teacher decoder

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import glob
import cv2

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

dir_name = "distillation_4"
img_name = "000000000036"
base_path = "/cluster/work/igp_psr/niacobone"
student = False
# frames_glob deve essere una lista, appaiata con dir_names
frames_glob = "*.jpg"

input_path = os.path.join(base_path, "examples/photos", dir_name)
print(f"Input path: {input_path}")
# output_path = os.path.join(base_path, "consistency_test", dir_name)
output_path = os.path.join(input_path, img_name)
os.makedirs(output_path, exist_ok=True)
frame_paths = sorted(glob.glob(os.path.join(input_path, frames_glob)))
print("Found frames:")
for p in frame_paths:
    print(f" - {p}")
frames = [np.array(Image.open(p).convert("RGB")) for p in frame_paths]

sam2_checkpoint = "/cluster/scratch/niacobone/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l_nico.yaml" # I set use_high_res_features_in_sam: false

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
# disabilito l'uso runtime (i layer restano caricati ma non vengono chiamati)
sam2.sam_mask_decoder.use_high_res_features = False
sam2.use_high_res_features_in_sam = False
sam2.num_feature_levels = 1

mask_generator_1 = SAM2AutomaticMaskGenerator(sam2)
mask_generator_2 = SAM2AutomaticMaskGenerator(sam2, pred_iou_thresh=0.0, stability_score_thresh=0.0)
mask_generator_3 = SAM2AutomaticMaskGenerator(sam2, pred_iou_thresh=0.4, stability_score_thresh=0.5)
mask_generator_4 = SAM2AutomaticMaskGenerator(sam2, pred_iou_thresh=0.4, stability_score_thresh=0.5, mask_threshold=0.2)
mask_generator_5 = SAM2AutomaticMaskGenerator(sam2, pred_iou_thresh=0.4, stability_score_thresh=0.5, mask_threshold=-0.2)

image = frames[0]
masks_1 = mask_generator_1.generate(image)
masks_2 = mask_generator_2.generate(image)
masks_3 = mask_generator_3.generate(image)
masks_4 = mask_generator_4.generate(image)
masks_5 = mask_generator_5.generate(image)

plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks_1)
plt.axis('off')
if student:
    output_file = os.path.join(output_path, "student_normal.png")
else:
    output_file = os.path.join(output_path, "teacher_normal.png")
plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
plt.close()
print(f"Saved to {output_file}")

plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks_2)
plt.axis('off')
if student:
    output_file = os.path.join(output_path, "student_no_threshold.png")
else:
    output_file = os.path.join(output_path, "teacher_no_threshold.png")
plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
plt.close()
print(f"Saved to {output_file}")

plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks_3)
plt.axis('off')
if student:
    output_file = os.path.join(output_path, "student_medium_threshold.png")
else:
    output_file = os.path.join(output_path, "teacher_medium_threshold.png")
plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
plt.close()
print(f"Saved to {output_file}")

plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks_4)
plt.axis('off')
if student:
    output_file = os.path.join(output_path, "student_medium_threshold_mask_threshold_02.png")
else:
    output_file = os.path.join(output_path, "teacher_medium_threshold_mask_threshold_02.png")
plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
plt.close()
print(f"Saved to {output_file}")

plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks_5)
plt.axis('off')
if student:
    output_file = os.path.join(output_path, "student_medium_threshold_mask_threshold_neg_02.png")
else:
    output_file = os.path.join(output_path, "teacher_medium_threshold_mask_threshold_neg_02.png")
plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
plt.close()
print(f"Saved to {output_file}")

# DEBUG - test rimozione filtri e abbassamento threshold per provare a vedere qualche segmentation mask con embedding additional head
# sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
# # disabilito l'uso runtime (i layer restano caricati ma non vengono chiamati)
# sam2.sam_mask_decoder.use_high_res_features = False
# sam2.use_high_res_features_in_sam = False
# sam2.num_feature_levels = 1

# mask_generator = SAM2AutomaticMaskGenerator(
#     sam2,
#     points_per_side=8,          # ridotto (64 punti)
#     multimask_output=False,     # 1 maschera per punto
#     pred_iou_thresh=0.0,
#     stability_score_thresh=0.0,
#     mask_threshold=-5.0,
#     min_mask_region_area=0,
#     box_nms_thresh=1.0,
#     crop_n_layers=0,
# )

# plt.figure(figsize=(20, 20))
# plt.imshow(image)
# show_anns(masks)
# plt.axis('off')
# output_file = os.path.join(output_path, "masks_consistency_coco_no_thresh_4999.png")
# # output_file = os.path.join(output_path, "masks_original.png")
# plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
# plt.close()
# print(f"Saved to {output_file}")