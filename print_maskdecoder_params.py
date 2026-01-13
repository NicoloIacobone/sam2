import torch
from sam2.build_sam import build_sam2

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

sam2_checkpoint = "/cluster/scratch/niacobone/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_decoder = sam2.sam_mask_decoder
total_params = sum(p.numel() for p in mask_decoder.parameters())
trainable_params = sum(p.numel() for p in mask_decoder.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
