from transformers import Mask2FormerForUniversalSegmentation
import timm, torch, re

hf_model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-base-IN21k-cityscapes-semantic"
)
hf_state = hf_model.model.pixel_level_module.encoder.state_dict()

def map_keys(k):
    k = k.replace("embeddings.patch_embeddings.projection.", "patch_embed.proj.")
    k = k.replace("embeddings.norm.", "patch_embed.norm.")
    k = re.sub(r"^encoder\.layers\.", "layers.", k)
    k = k.replace(".layernorm_before.", ".norm1.")
    k = k.replace(".layernorm_after.", ".norm2.")
    k = k.replace(".attention.self.relative_position_bias_table", ".attn.relative_position_bias_table")
    k = k.replace(".attention.output.dense.", ".attn.proj.")
    k = k.replace(".intermediate.dense.", ".mlp.fc1.")
    k = k.replace(".output.dense.", ".mlp.fc2.")
    k = re.sub(r"^encoder\.layernorm\.", "norm.", k)
    return k

merged = {}
skip = set()

for k in list(hf_state.keys()):

    if k in skip:
        continue
    if "relative_position_index" in k:
        continue
    if ".attention.self.query." in k:
        k_k = k.replace(".attention.self.query.", ".attention.self.key.")
        k_v = k.replace(".attention.self.query.", ".attention.self.value.")
        qkv_key = map_keys(k).replace(".attention.self.query.", ".attn.qkv.")
        merged[qkv_key] = torch.cat([hf_state[k], hf_state[k_k], hf_state[k_v]], dim=0)
        skip.update([k, k_k, k_v])
        continue

    merged[map_keys(k)] = hf_state[k]

timm_model = timm.create_model('swin_base_patch4_window7_224', pretrained=False)
timm_keys = set(timm_model.state_dict().keys())
mapped_keys = set(merged.keys())
matched   = mapped_keys & timm_keys
unmatched = mapped_keys - timm_keys
missing   = timm_keys - mapped_keys

print(f"\nCorrectly mapped: {len(matched)} / {len(timm_keys)}")
print("\nMissing (7 expected, all timm-specific):")
for val in sorted(missing):
    print(val)
print("\nUnmatched (expected, all hugging-face specific):")
for val in sorted(unmatched):
    print(val)

if len(matched) == 322:
    torch.save(merged, "./pretrain/mask2former-swinb-cityscapes.pth")
    print("\n322 parameters saved for cityscapes semantic segmentation.")
else:
    print(f"\nUnexpected number of parameters - path file not saved.")