import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import glob
import pickle

from sam2.build_sam import build_sam2_video_predictor

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

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

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

base_path = "/cluster/work/igp_psr/niacobone"
dir_name = "tenda_ufficio_sam"
input_path = os.path.join(base_path, "examples/photos", dir_name)

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(input_path)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

sam2_checkpoint = "/cluster/scratch/niacobone/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

inference_state = predictor.init_state(video_path=input_path)

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# Let's add a positive click at (x, y) = (210, 350) to get started
points = np.array([[2500, 1800], [1900, 1900]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

out_file = os.path.join(input_path, "video_segments.npy")
np.save(out_file, video_segments, allow_pickle=True)
print(f"video_segments salvato in {out_file}")

# input_path = os.path.join(base_path, "examples/photos", dir_name)
# output_path = os.path.join(base_path, "automatic_mask_generator", dir_name)
# os.makedirs(output_path, exist_ok=True)
# frame_paths = sorted(glob.glob(os.path.join(input_path, frames_glob)))
# frames = [np.array(Image.open(p).convert("RGB")) for p in frame_paths]


# sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
# # disabilito l'uso runtime (i layer restano caricati ma non vengono chiamati)
# sam2.sam_mask_decoder.use_high_res_features = False
# sam2.use_high_res_features_in_sam = False
# sam2.num_feature_levels = 1

# mask_generator = SAM2AutomaticMaskGenerator(sam2)
# # per questo test considero solo la prima immagine
# image = frames[0]
# masks = mask_generator.generate(image)

# # Save the masks object as a numpy file
# np.save(os.path.join(output_path, "masks.npy"), masks)
# print(f"Masks saved to {os.path.join(output_path, 'masks.npy')}")

# plt.figure(figsize=(20, 20))
# plt.imshow(image)
# show_anns(masks)
# plt.axis('off')
# output_file = os.path.join(output_path, "masks_consistency_coco.png")
# plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
# plt.close()
# print(f"Saved to {output_file}")

# # DEBUG - test rimozione filtri e abbassamento threshold per provare a vedere qualche segmentation mask con embedding additional head
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
# output_file = os.path.join(output_path, "masks_consistency_coco_no_thresh.png")
# # output_file = os.path.join(output_path, "masks_original.png")
# plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
# plt.close()
# print(f"Saved to {output_file}")