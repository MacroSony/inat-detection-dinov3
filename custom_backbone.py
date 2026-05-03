import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure dinov3 is in the python path
sys.path.append(os.path.join(os.path.dirname(__file__), "dinov3"))

from dinov3.models.vision_transformer import vit_small

# Import RF-DETR backbone components
from rfdetr.models.backbone.base import BackboneBase
from rfdetr.models.backbone.projector import MultiScaleProjector
from rfdetr.utilities.tensors import NestedTensor

class DinoV3EncoderWrapper(nn.Module):
    def __init__(self, weights_path, out_feature_indexes=[2, 4, 5, 9]):
        super().__init__()
        
        # Initialize the native DINOv3 ViT-S/16 model
        print(f"Instantiating DINOv3 ViT-S/16 backbone...")
        self.vit = vit_small(
            patch_size=16,
            n_storage_tokens=4,
            mask_k_bias=True,
            layerscale_init=1e-5
        )
        
        # Load local weights
        print(f"Loading local weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # Handle dict nesting if present
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'student' in state_dict:
            state_dict = state_dict['student']
            
        self.vit.load_state_dict(state_dict, strict=True)
        print("DINOv3 weights loaded successfully!")
        
        self.out_feature_indexes = out_feature_indexes
        
        # ViT-S embedding dimension is 384
        self._out_feature_channels = [384] * len(out_feature_indexes)
        
    def forward(self, x):
        # get_intermediate_layers can take a list of indices
        # reshape=True will reshape the sequence back to spatial (B, C, H, W)
        features = self.vit.get_intermediate_layers(
            x, 
            n=self.out_feature_indexes,
            reshape=True,
            norm=True,
            return_class_token=False,
            return_extra_tokens=False
        )
        return features

class CustomDinoV3Backbone(BackboneBase):
    def __init__(
        self,
        weights_path: str,
        out_channels: int = 256,
        out_feature_indexes: list = [2, 4, 5, 9],
        projector_scale: list = ["P3", "P4", "P5"],
        layer_norm: bool = False,
        rms_norm: bool = False,
        freeze_encoder: bool = False
    ):
        super().__init__()
        
        self.encoder = DinoV3EncoderWrapper(weights_path, out_feature_indexes)
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        # Build the MultiScaleProjector just like RF-DETR
        level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
        scale_factors = [level2scalefactor[lvl] for lvl in projector_scale]
        
        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
        )
        
    def forward(self, tensor_list: NestedTensor):
        # 1. Pass image batch through DINOv3 (B, C, H, W)
        feats = self.encoder(tensor_list.tensors)
        
        # 2. Pass feature pyramid through projector
        feats = self.projector(feats)
        
        # 3. Create NestedTensors with downsampled masks
        out = []
        for feat in feats:
            m = tensor_list.mask
            assert m is not None
            # Interpolate the boolean mask down to the feature map resolution
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(feat, mask))
            
        return out
        
    def get_named_param_lr_pairs(self, args, prefix: str = "backbone.0"):
        # For a 12-layer ViT-S
        num_layers = args.out_feature_indexes[-1] + 1
        backbone_key = "backbone.0.encoder"
        named_param_lr_pairs = {}
        
        # Fallback simplistic learning rate decay strategy similar to RF-DETR
        for n, p in self.named_parameters():
            full_n = prefix + "." + n
            if backbone_key in full_n and p.requires_grad:
                # Basic learning rate assignment for the encoder
                lr = args.lr_encoder
                wd = args.weight_decay
                named_param_lr_pairs[full_n] = {
                    "params": p,
                    "lr": lr,
                    "weight_decay": wd,
                }
        return named_param_lr_pairs

# --- Monkey-Patching RF-DETR Registry ---
import rfdetr.models.backbone.backbone as rfdetr_backbone_module
import rfdetr.models.backbone as rfdetr_backbone_init

OriginalBackbone = rfdetr_backbone_module.Backbone

def build_custom_backbone_or_original(
    name: str,
    pretrained_encoder: str = None,
    window_block_indexes: list = None,
    drop_path: float = 0.0,
    out_channels: int = 256,
    out_feature_indexes: list = None,
    projector_scale: list = None,
    use_cls_token: bool = False,
    freeze_encoder: bool = False,
    layer_norm: bool = False,
    target_shape: tuple = (640, 640),
    rms_norm: bool = False,
    backbone_lora: bool = False,
    gradient_checkpointing: bool = False,
    load_dinov2_weights: bool = True,
    patch_size: int = 14,
    num_windows: int = 4,
    positional_encoding_size: int = 0,
):
    if name == "dinov3_vits16":
        print("Injecting Custom DINOv3 Backbone!")
        # We assume the weights are at this path
        weights_path = "models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
        return CustomDinoV3Backbone(
            weights_path=weights_path,
            out_channels=out_channels,
            out_feature_indexes=out_feature_indexes,
            projector_scale=projector_scale,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
            freeze_encoder=freeze_encoder
        )
    return OriginalBackbone(
        name=name,
        pretrained_encoder=pretrained_encoder,
        window_block_indexes=window_block_indexes,
        drop_path=drop_path,
        out_channels=out_channels,
        out_feature_indexes=out_feature_indexes,
        projector_scale=projector_scale,
        use_cls_token=use_cls_token,
        freeze_encoder=freeze_encoder,
        layer_norm=layer_norm,
        target_shape=target_shape,
        rms_norm=rms_norm,
        backbone_lora=backbone_lora,
        gradient_checkpointing=gradient_checkpointing,
        load_dinov2_weights=load_dinov2_weights,
        patch_size=patch_size,
        num_windows=num_windows,
        positional_encoding_size=positional_encoding_size,
    )

def apply_monkey_patch():
    rfdetr_backbone_module.Backbone = build_custom_backbone_or_original
    rfdetr_backbone_init.Backbone = build_custom_backbone_or_original
    print("Monkey-patch applied: `Backbone` now supports 'dinov3_vits16'")
