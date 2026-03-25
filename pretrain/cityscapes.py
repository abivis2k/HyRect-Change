from transformers import Mask2FormerForUniversalSegmentation
import torch
from timm import create_model

def map_key(k):
    k = k.replace("encoder.", "")
    k = k.replace("attention.", "attn.")
    k = k.replace("layernorm", "norm")
    return k

def load_cityscapes():
    timm_swin = create_model('swin_base_patch4_window7_224', pretrained=False, features_only=True, img_size=256)
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-IN21k-cityscapes-semantic")
    backbone = model.model.pixel_level_module.encoder
    cc_state = backbone.state_dict()
    timm_state = timm_swin.state_dict()

    new_state = {}
    num_matches = 0
    new_state["patch_embed.proj.weight"] = cc_state["embeddings.patch_embeddings.projection.weight"]
    new_state["patch_embed.proj.bias"]   = cc_state["embeddings.patch_embeddings.projection.bias"]
    new_state["patch_embed.norm.weight"] = cc_state["embeddings.norm.weight"]
    new_state["patch_embed.norm.bias"]   = cc_state["embeddings.norm.bias"]
    num_matches += 4

    for k in cc_state:
        if "encoder.layers" not in k:
            continue
        k2 = k.replace("encoder.layers.", "layers_")
        k2 = k2.replace("layernorm_before", "norm1")
        k2 = k2.replace("layernorm_after", "norm2")
        k2 = k2.replace("intermediate.dense", "mlp.fc1")
        k2 = k2.replace("output.dense", "mlp.fc2")
        k2 = k2.replace("attention.output.dense", "attn.proj")
        k2 = k2.replace("attention.self.relative_position_bias_table", "attn.relative_position_bias_table")

        if any(x in k for x in ["query", "key", "value"]):
            continue

        if k2 in timm_state and cc_state[k].shape == timm_state[k2].shape:
            new_state[k2] = cc_state[k]
            num_matches += 1

    for layer in range(4): 
        for block in range(2 * (2 ** layer)):  
            base = f"encoder.layers.{layer}.blocks.{block}.attention.self."

            try:
                q_w = cc_state[base + "query.weight"]
                k_w = cc_state[base + "key.weight"]
                v_w = cc_state[base + "value.weight"]

                q_b = cc_state[base + "query.bias"]
                k_b = cc_state[base + "key.bias"]
                v_b = cc_state[base + "value.bias"]
            except:
                continue

            qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
            qkv_b = torch.cat([q_b, k_b, v_b], dim=0)

            new_key_w = f"layers_{layer}.blocks.{block}.attn.qkv.weight"
            new_key_b = f"layers_{layer}.blocks.{block}.attn.qkv.bias"

            if new_key_w in timm_state:
                new_state[new_key_w] = qkv_w
                new_state[new_key_b] = qkv_b
                num_matches += 2

    print(f"Number of matched layers: {num_matches}")
    timm_state.update(new_state)
    timm_swin.load_state_dict(timm_state, strict=False)
    torch.save(timm_swin.state_dict(), "./pretrain/mask2former-swin-base-cityscapes-semantic.pth")
    print("File saved...")

load_cityscapes()