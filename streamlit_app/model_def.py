"""Model definitions for the PG-MMIL shoplifting detector.

The classes below are copied verbatim from the research notebook
(Shoplifting_PG_MMIL_Research_Final results.ipynb, Phase 5) so that checkpoints
saved by the notebook load with identical parameter names and shapes.

The Sultani-, RTFM-, Shopformer- and VadCLIP-named models are compact
*inspired* baselines on the cached 567-D features, not exact reproductions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def compute_skeletal_kinematics(pose_raw):
    """
    pose_raw: [B, T, 51] -> 17 keypoints x (x, y, conf)
    Computes hip-centered coordinates, 1st-order velocity, and 2nd-order acceleration -> [B, T, 153]
    """
    B, T, _ = pose_raw.shape
    kpts = pose_raw.view(B, T, 17, 3)
    xy = kpts[..., :2]
    conf = kpts[..., 2:3]

    # Hip-center coordinates (COCO indices 11 & 12 are left/right hip) for translation invariance
    hip_center = 0.5 * (xy[:, :, 11:12, :] + xy[:, :, 12:13, :])
    xy_rel = (xy - hip_center) * (conf > 0.1).float()
    pose_centered = torch.cat([xy_rel, conf], dim=-1).view(B, T, 51)

    # 1st-order Temporal Velocity (dP/dt)
    vel = torch.zeros_like(pose_centered)
    vel[:, 1:, :] = pose_centered[:, 1:, :] - pose_centered[:, :-1, :]

    # 2nd-order Temporal Acceleration (d^2P/dt^2 - captures sudden hand-to-pocket snatching)
    accel = torch.zeros_like(pose_centered)
    accel[:, 2:, :] = vel[:, 2:, :] - vel[:, 1:-1, :]

    return torch.cat([pose_centered, vel, accel], dim=-1) # [B, T, 153]

# --- BASELINE 1: Sultani et al. (Standard WS-VAD MIL) ---
class SultaniMIL(nn.Module):
    def __init__(self, num_domains=4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.domain_head = nn.Linear(64, num_domains)

    def forward(self, x, alpha=0.0):
        vis_raw = x[:, :, 51:563]
        feat = self.backbone(vis_raw)
        scores = self.classifier(feat).squeeze(-1)
        dom_logits = self.domain_head(feat.mean(dim=1))
        return scores, feat, dom_logits

# --- BASELINE 2: RTFM (Pyramid Dilated Temporal Conv + Magnitude MIL) ---
class RTFMNet(nn.Module):
    def __init__(self, num_domains=4):
        super().__init__()
        self.conv1 = nn.Conv1d(512, 128, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(512, 128, kernel_size=3, padding=2, dilation=2)
        self.norm = nn.LayerNorm(256)
        self.attn = nn.MultiheadAttention(256, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())
        self.domain_head = nn.Linear(256, num_domains)

    def forward(self, x, alpha=0.0):
        vis = x[:, :, 51:563].transpose(1, 2) # [B, 512, T]
        h = torch.cat([F.gelu(self.conv1(vis)), F.gelu(self.conv2(vis))], dim=1).transpose(1, 2) # [B, T, 256]
        attn_out, _ = self.attn(h, h, h)
        feat = self.norm(h + attn_out)
        scores = self.classifier(feat).squeeze(-1)
        dom_logits = self.domain_head(feat.mean(dim=1))
        return scores, feat, dom_logits

# --- BASELINE 3: Shopformer / Pose-Only Kinematic Transformer ---
class ShopformerPoseMIL(nn.Module):
    def __init__(self, num_domains=4):
        super().__init__()
        self.pose_emb = nn.Sequential(
            nn.Linear(153, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        enc_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())
        self.domain_head = nn.Linear(128, num_domains)

    def forward(self, x, alpha=0.0):
        kin = compute_skeletal_kinematics(x[:, :, :51])
        feat = self.transformer(self.pose_emb(kin))
        scores = self.classifier(feat).squeeze(-1)
        dom_logits = self.domain_head(feat.mean(dim=1))
        return scores, feat, dom_logits

# --- BASELINE 4: VadCLIP Visual-Prompt MIL ---
class VadCLIPVisualPromptMIL(nn.Module):
    def __init__(self, num_domains=4):
        super().__init__()
        self.vis_proj = nn.Sequential(
            nn.Linear(516, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.25)
        )
        enc_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())
        self.domain_head = nn.Linear(128, num_domains)

    def forward(self, x, alpha=0.0):
        vis_raw = x[:, :, 51:563]
        prompts = x[:, :, 563:]
        suspicion_gate = torch.sigmoid(prompts[:, :, :3].sum(dim=-1, keepdim=True) - prompts[:, :, 3:4])
        v_in = torch.cat([vis_raw * (1.0 + suspicion_gate), prompts], dim=-1)
        feat = self.transformer(self.vis_proj(v_in))
        scores = self.classifier(feat).squeeze(-1)
        dom_logits = self.domain_head(feat.mean(dim=1))
        return scores, feat, dom_logits

# --- PROPOSED ARCHITECTURE: Cross-Domain PG-MMIL (KinematicSemanticMIL) ---
class ProposedPGMMIL(nn.Module):
    """PG-MMIL with optional switches for controlled component ablations."""
    def __init__(self, num_domains=4, use_grl=True, use_pose=True, use_visual=True,
                 use_prompts=True, use_cross_attention=True, use_gate=True):
        super().__init__()
        self.use_grl = use_grl
        self.use_pose = use_pose
        self.use_visual = use_visual
        self.use_prompts = use_prompts
        self.use_cross_attention = use_cross_attention
        self.use_gate = use_gate
        self.pose_proj = nn.Sequential(nn.Linear(153, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2))
        pose_enc = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.2, batch_first=True)
        self.pose_transformer = nn.TransformerEncoder(pose_enc, num_layers=2)
        self.vis_proj = nn.Sequential(nn.Linear(516, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.25))
        vis_enc = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.2, batch_first=True)
        self.vis_transformer = nn.TransformerEncoder(vis_enc, num_layers=2)
        self.p2v_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, dropout=0.1, batch_first=True)
        self.v2p_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, dropout=0.1, batch_first=True)
        self.gate_net = nn.Sequential(nn.Linear(256, 256), nn.Sigmoid())
        self.fusion_norm = nn.LayerNorm(256)
        self.anomaly_head = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1), nn.Sigmoid())
        self.domain_head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, num_domains))

    def forward(self, x, alpha=0.1):
        batch, steps, _ = x.shape
        pose_raw, vis_raw, prompts = x[:, :, :51], x[:, :, 51:563], x[:, :, 563:]
        if self.use_pose:
            kin = compute_skeletal_kinematics(pose_raw)
            p_feat = self.pose_transformer(self.pose_proj(kin))
        else:
            p_feat = torch.zeros(batch, steps, 128, device=x.device, dtype=x.dtype)
        if self.use_visual:
            prompt_input = prompts if self.use_prompts else torch.zeros_like(prompts)
            if self.use_prompts:
                anomaly_prior = torch.sigmoid(prompt_input[:, :, :3].max(dim=-1, keepdim=True)[0] - prompt_input[:, :, 3:4])
                visual_input = vis_raw * (1.0 + 0.5 * anomaly_prior)
            else:
                visual_input = vis_raw
            v_feat = self.vis_transformer(self.vis_proj(torch.cat([visual_input, prompt_input], dim=-1)))
        else:
            v_feat = torch.zeros(batch, steps, 128, device=x.device, dtype=x.dtype)
        if self.use_cross_attention:
            p_attended, _ = self.p2v_attn(query=p_feat, key=v_feat, value=v_feat)
            v_attended, _ = self.v2p_attn(query=v_feat, key=p_feat, value=p_feat)
            p_out, v_out = p_feat + p_attended, v_feat + v_attended
        else:
            p_out, v_out = p_feat, v_feat
        concat_feat = torch.cat([p_out, v_out], dim=-1)
        if self.use_gate:
            gate = self.gate_net(concat_feat)
            fused = self.fusion_norm(concat_feat * gate)
        else:
            fused = self.fusion_norm(concat_feat)
        scores = self.anomaly_head(fused).squeeze(-1)
        pooled_feat = fused.mean(dim=1)
        effective_alpha = alpha if self.use_grl else 0.0
        rev_feat = GradientReversalFn.apply(pooled_feat, effective_alpha)
        domain_logits = self.domain_head(rev_feat)
        return scores, fused, domain_logits


# name -> (factory, top-k used for the bag score, human label)
MODEL_REGISTRY = {
    "Proposed_PG_MMIL_Full":   (lambda: ProposedPGMMIL(4, use_grl=True),  3, "PG-MMIL (full, with GRL)"),
    "Proposed_PG_MMIL_NoGRL":  (lambda: ProposedPGMMIL(4, use_grl=False), 3, "PG-MMIL (no GRL)"),
    "VadCLIP_VisualPromptMIL": (lambda: VadCLIPVisualPromptMIL(4),        3, "VadCLIP-inspired prompt MIL"),
    "RTFM_Net":                (lambda: RTFMNet(4),                       3, "RTFM-inspired TCN"),
    "Shopformer_PoseMIL":      (lambda: ShopformerPoseMIL(4),             3, "Pose-only Transformer"),
    "Sultani_MIL":             (lambda: SultaniMIL(4),                    1, "Sultani-inspired MIL"),
}
