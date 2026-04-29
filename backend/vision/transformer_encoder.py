"""
Vision Transformer Encoder for Robotic Grasping
Processes RGB-D images using ViT/DINOv2 to extract spatial features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
import timm


class RGBDEncoder(nn.Module):
    """
    Dual-branch RGB-D encoder.
    RGB → ViT Transformer
    Depth → Dedicated depth encoder (hashed into bins)
    """
    
    def __init__(
        self,
        d_model: int = 768,
        image_size: int = 224,
        patch_size: int = 16,
        rgb_pretrained: bool = True,
        depth_channels: int = 1,
        use_dinov2: bool = False,
        freeze_vit: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.image_size = image_size
        self.patch_size = patch_size
        self.use_dinov2 = use_dinov2
        
        # === RGB Branch (ViT) ===
        if use_dinov2:
            # DINOv2 - self-supervised, better features
            self.vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            d_backbone = 768
        else:
            # Supervised ViT
            self.vit = ViTModel(ViTConfig(
                image_size=image_size,
                patch_size=patch_size,
                hidden_size=d_model,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=d_model * 4
            ))
            d_backbone = d_model
        
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # === Depth Branch ===
        # Depth → 64-bin histogram → embedding
        self.depth_bins = 64
        self.depth_embedding = nn.Embedding(self.depth_bins, d_model // 4)
        
        # Process depth as 1-channel image with dedicated CNN
        self.depth_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten()
        )  # Output: 128 * 7 * 7 = 6272
        
        self.depth_proj = nn.Sequential(
            nn.Linear(6272, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # === Fusion ===
        self.fusion = nn.Sequential(
            nn.Linear(d_backbone + d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # === State Encoder (robot joint positions + velocities) ===
        self.state_encoder = nn.Sequential(
            nn.Linear(14, 256),  # 7 joints: pos(7) + vel(7)
            nn.ReLU(),
            nn.Linear(256, d_model),
            nn.LayerNorm(d_model)
        )
        
        # === Cross-attention: vision attends to robot state ===
        self.cross_attention = nn.MultiheadAttention(
            d_model, n_heads=8, dropout=0.1, batch_first=True
        )
        
    def forward(self, rgb_image, depth_image, robot_state):
        """
        Args:
            rgb_image: (B, 3, H, W) — RGB image
            depth_image: (B, 1, H, W) — Depth map
            robot_state: (B, 14) — joint positions(7) + velocities(7)
        Returns:
            visual_features: (B, d_model)
        """
        # === RGB Encoding ===
        if self.use_dinov2:
            with torch.no_grad():
                rgb_features = self.vit.forward_features(rgb_image)
                # Use CLS token features
                rgb_emb = rgb_features[:, 0]  # (B, 768)
        else:
            rgb_outputs = self.vit(rgb_image)
            rgb_emb = rgb_outputs.last_hidden_state[:, 0]  # (B, d_model)
        
        # === Depth Encoding ===
        # Discretize depth into bins
        depth_bins = torch.bucketize(
            depth_image, 
            torch.linspace(0, 1, self.depth_bins - 1, device=depth_image.device)
        )  # (B, 1, H, W)
        depth_bins = depth_bins.squeeze(1).long()  # (B, H, W)
        
        # Flatten and embed
        B, H, W = depth_bins.shape
        depth_flat = depth_bins.reshape(B, -1)  # (B, H*W)
        
        # Embed each depth pixel
        depth_emb_flat = self.depth_embedding(depth_flat)  # (B, H*W, d_model//4)
        
        # Reshape to 2D, pool
        depth_h = H // 8
        depth_w = W // 8
        depth_emb = depth_emb_flat.reshape(B, depth_h, depth_w, -1).permute(0, 3, 1, 2)
        depth_emb = F.adaptive_avg_pool2d(depth_emb, (1, 1)).flatten(1)  # (B, d_model//4)
        depth_emb = self.depth_proj(depth_emb)  # (B, d_model)
        
        # === Robot State Encoding ===
        state_emb = self.state_encoder(robot_state)  # (B, d_model)
        
        # === Fuse RGB + Depth ===
        visual_emb = torch.cat([rgb_emb, depth_emb], dim=-1)  # (B, d_model * 2)
        visual_emb = self.fusion(visual_emb)  # (B, d_model)
        
        # === Cross-attention between visual features and robot state ===
        # Visual features attend to robot state
        visual_query = visual_emb.unsqueeze(1)  # (B, 1, d_model)
        state_key = state_emb.unsqueeze(1)  # (B, 1, d_model)
        
        attended, attn_weights = self.cross_attention(
            query=visual_query,
            key=state_key,
            value=state_key
        )
        
        # Combine original + attended
        visual_features = visual_emb + attended.squeeze(1)  # Residual
        
        return visual_features, state_emb, attn_weights


class SpatialAttention(nn.Module):
    """
    Spatial attention for focusing on grasp-relevant regions.
    """
    
    def __init__(self, d_model: int = 768):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )
        
    def forward(self, visual_features, rgb_image):
        """
        Args:
            visual_features: (B, d_model)
            rgb_image: (B, 3, H, W)
        Returns:
            attended_features: (B, d_model)
            spatial_weights: (B, H//patch, W//patch)
        """
        B, C, H, W = rgb_image.shape
        patch_h = H // 16
        patch_w = W // 16
        
        # This would need patch-level features from ViT
        # For now, return original
        return visual_features, None


class GraspQualityNetwork(nn.Module):
    """
    Predicts grasp success probability given visual features + action.
    Used for grasp ranking and refinement.
    """
    
    def __init__(self, d_model: int = 768, n_actions: int = 4):
        super().__init__()
        
        # Action: [delta_x, delta_y, delta_z, delta_gripper]
        self.action_encoder = nn.Linear(n_actions, d_model)
        
        # Grasp predictor
        self.grasp_net = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),  # visual + state + action
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, visual_features, state_features, action):
        """
        Args:
            visual_features: (B, d_model)
            state_features: (B, d_model)
            action: (B, 4) — [dx, dy, dz, dgripper]
        Returns:
            grasp_quality: (B, 1) — success probability 0-1
        """
        action_emb = self.action_encoder(action)  # (B, d_model)
        
        combined = torch.cat([visual_features, state_features, action_emb], dim=-1)
        quality = self.grasp_net(combined)  # (B, 1)
        
        return quality
