from transformers import Mask2FormerForUniversalSegmentation
import torch

model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-base-IN21k-cityscapes-semantic"
)
backbone_state = model.model.pixel_level_module.encoder.state_dict()

torch.save(backbone_state, "./pretrain/mask2former-swinb-cityscapes.pth")

print(f"Saved {len(backbone_state)} parameters for backbone.")