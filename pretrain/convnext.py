import timm
import torch

model = timm.create_model('convnextv2_base', pretrained=True, features_only=True)
torch.save(model.state_dict(), './pretrain/convnextv2_base.pth')