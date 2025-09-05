import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

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

def main():
    parser = argparse.ArgumentParser(description="SAM2 Mask Generator Example")
    parser.add_argument('--input_dir', type=str, default='bouncing_balls', help='Nome cartella immagini di input (default: input_images)')
    # parser.add_argument('--output_dir', type=str, default='/cluster/work/igp_psr/niacobone/examples/kubric', help='Cartella di output per i risultati')
    args = parser.parse_args()

    input_dir = '/cluster/work/igp_psr/niacobone/examples/kubric/' + args.input_dir
    output_dir = '/cluster/work/igp_psr/niacobone/examples/kubric/results/sam2/' + args.input_dir

    # Selezione device
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
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    os.makedirs(output_dir, exist_ok=True)
    # Filtra solo immagini comuni
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    for fname in os.listdir(input_dir):
        if os.path.splitext(fname)[1].lower() in valid_exts:
            process_image(os.path.join(input_dir, fname), mask_generator, output_dir)

if __name__ == "__main__":
    main()