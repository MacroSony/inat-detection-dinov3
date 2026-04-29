import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "rf-detr", "src"))
from custom_backbone import CustomDinoV3Backbone, apply_monkey_patch
from rfdetr.utilities.tensors import NestedTensor

apply_monkey_patch()

# Initialize the custom backbone
weights_path = "/home/bruhw/programming/inat-detection-dinov3/models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
backbone = CustomDinoV3Backbone(weights_path=weights_path)

# Create a dummy image tensor (1, 3, 640, 640)
x = torch.randn(1, 3, 640, 640)
mask = torch.zeros(1, 640, 640, dtype=torch.bool)
tensor_list = NestedTensor(x, mask)

print("Running forward pass...")
with torch.no_grad():
    out = backbone(tensor_list)

print("Output features:")
for i, feat in enumerate(out):
    print(f"Layer {i}: Shape {feat.tensors.shape}")

print("Test passed!")
