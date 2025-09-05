print("importing libraries...")
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import re
print("libraries imported.")

def show_anns_on_ax(ax, anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
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
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)
    ax.imshow(img)

def process_image(image_path, mask_generator, output_dir):
    image = Image.open(image_path)
    image_np = np.array(image.convert("RGB"))
    masks = mask_generator.generate(image_np)
    print(f"{os.path.basename(image_path)}: {len(masks)} maschere trovate")
    # Visualizza e salva il risultato
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image_np)
    show_anns_on_ax(ax, masks)
    ax.axis('off')
    out_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_masks.png")
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Result saved to: {out_path}")

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

def main():
    parser = argparse.ArgumentParser(description="SAM2 Mask Generator Example")
    parser.add_argument('--input_dir', type=str, default='bouncing_balls', help='Nome cartella immagini di input (default: input_images)')
    # parser.add_argument('--output_dir', type=str, default='/cluster/work/igp_psr/niacobone/examples/kubric', help='Cartella di output per i risultati')
    parser.add_argument('--video', type=bool, default=False, help='If True, process video instead of images (default: False)')
    args = parser.parse_args()

    input_dir = '/cluster/work/igp_psr/niacobone/examples/kubric/' + args.input_dir
    output_dir = '/cluster/work/igp_psr/niacobone/examples/kubric/results/sam2/' + args.input_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Selezione device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    print("Loading SAM2 model and mask generator...")
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    if args.video:
        print("Video mode enabled.")
        sam2 = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

        # scan all the frame names in this directory
        frame_names = [
            p for p in os.listdir(input_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        # frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        frame_names.sort(key=lambda p: int(re.search(r'\d+$', os.path.splitext(p)[0]).group()))

        # initialize an inference state (loads all frames and stores them in inference_state)
        inference_state = sam2.init_state(video_path=input_dir)
        print(f"Initialized inference state with {len(frame_names)} frames.")

        # SOSTITUISCI QUI SOTTO DOPO AVER SCRITTO LO SCRIPT CHE INIZIALIZZA LE MASCHERE
        ann_frame_idx = 0 # the frame index we interact with
        ann_obj_id = 1 # give a unique id to each object we interact with (it can be any integers)

        # Let's add a positive click at (x, y) = (210, 350) to get started
        points = np.array([[129, 202], [323, 139], [265, 365], [187, 399]], dtype=np.float32)
        # for labels, `1` means positive click and `0` means negative click
        labels = np.array([1, 1, 1, 1], np.int32)
        _, out_obj_ids, out_mask_logits = sam2.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

        # propagate the prompts to get the masklet across the video
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in sam2.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # save the segmentation results every few frames
        vis_frame_stride = 1 # loop over all frames
        # Loop through frames at intervals defined by vis_frame_stride
        for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            # Load the current frame as a PIL image and convert to numpy array
            img = Image.open(os.path.join(input_dir, frame_names[out_frame_idx])).convert("RGB")
            img_np = np.array(img)
            # Retrieve the segmentation masks for the current frame
            masks = video_segments[out_frame_idx]
            # Create a matplotlib figure for visualization
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(img_np)
            # Overlay each mask on the image
            for out_obj_id, out_mask in masks.items():
                show_mask(out_mask, ax, obj_id=out_obj_id)
            ax.axis('off')
            # Save the visualization to the output directory
            save_path = os.path.join(output_dir, f"frame_{out_frame_idx:04d}_masks.png")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        print(f"Saved mask visualization to: {save_path}")
    else:
        print("Image mode enabled.")
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
        mask_generator = SAM2AutomaticMaskGenerator(sam2)
        print("Model and mask generator loaded.")

        # Filtra solo immagini comuni
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [fname for fname in os.listdir(input_dir) if os.path.splitext(fname)[1].lower() in valid_exts]
        print(f"Found {len(image_files)} valid image(s) in input directory.")

        for idx, fname in enumerate(image_files):
            print(f"Processing image {idx+1}/{len(image_files)}: {fname}")
            process_image(os.path.join(input_dir, fname), mask_generator, output_dir)
        print("Processing complete.")

if __name__ == "__main__":
    main()