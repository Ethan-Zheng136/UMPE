# se2_aligner.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SE2Aligner(nn.Module):
    """
    学习一个小的 SE(2) 微对齐： prior_map -> warp -> prior_map_aligned
    输入:
      bev_map   : [B, C, H, W]   参考 (来自主干 BEV)
      prior_map : [B, C, H, W]   要对齐的先验特征 (sat 或 rasterSD)
      cond      : [B, D] or None 条件(可用 source embedding/FiLM全局描述)
    输出:
      prior_aligned: [B, C, H, W]
      T_params     : dict{'dx_m','dy_m','dtheta','dx_norm','dy_norm','theta_mat':[B,2,3]}
      aux          : 可选辅助量(对齐损失项)
    备注:
      - dx,dy 的正方向默认与 "图像坐标" 一致: +x 向右，+y 向下。
      - 若你的 BEV 定义是 x 前进、y 向左，可用 flip_y/x 参数调整符号（见初始化参数）。
    """
    def __init__(self,
                 c=256, H=200, W=100,
                 m_per_cell=(0.3, 0.3),         # 每像素米数 (mx, my)
                 max_trans_m=(1.0, 1.0),        # 限制微平移幅度（米）
                 max_rot_deg=5.0,               # 限制微旋转幅度（度）
                 cond_dim=0,                     # 若>0，使用条件向量
                 use_corr=False,                 # 可选: 拼接相关性特征
                 flip_x=False, flip_y=False):    # 若你的 BEV 坐标系与图像坐标不同
        super().__init__()
        self.C, self.H, self.W = c, H, W
        self.mx, self.my = m_per_cell
        self.max_dx, self.max_dy = max_trans_m
        self.max_rot = math.radians(max_rot_deg)
        self.flip_x = -1.0 if flip_x else 1.0
        self.flip_y = -1.0 if flip_y else 1.0
        in_dim = 2*c
        if cond_dim > 0: in_dim += cond_dim
        if use_corr:
            # 简单“通道相关性”作全局提示：<bev, prior> 的 GAP 后 inner-product 向量
            self.corr_pool = nn.AdaptiveAvgPool2d((1,1))
            in_dim += c

        self.head = nn.Sequential(
            nn.Conv2d(2*c, c, 1, bias=False),   # 先把 [bev|prior] 融合到 C
            nn.GroupNorm(16, c), nn.ReLU(True)
        )
        # 全局聚合 + MLP 预测 (dx, dy, dtheta)
        mlp_in = c + (cond_dim if cond_dim>0 else 0) + (c if use_corr else 0)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 128), nn.ReLU(True),
            nn.Linear(128, 3)
        )
        # 零初始化最后一层，开局输出 (0,0,0) ≈ 恒等变换
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.use_corr = use_corr
        self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, bev_map, prior_map, cond=None, mode='bilinear', align_corners=False):
        """
        bev_map:   [B,C,H,W]
        prior_map: [B,C,H,W]
        cond:      [B,D] or None
        mode:      'bilinear' or 'nearest'
        """
        B, C, H, W = prior_map.shape
        assert (C==self.C and H==self.H and W==self.W), "size mismatch with init"

        # 1) 提取融合上下文 (空间 C×H×W → C×1×1 → 展平)
        x = torch.cat([bev_map, prior_map], dim=1)         # [B,2C,H,W]
        x = self.head(x)                                   # [B,C,H,W]
        g = self.gap(x).flatten(1)                         # [B,C]
        if cond is not None:
            g = torch.cat([g, cond], dim=-1)               # [B,C+D]
        if self.use_corr:
            # 简单相关性: GAP(bev*prior) 作为相关特征
            corr = self.corr_pool(bev_map * prior_map).flatten(1)  # [B,C]
            g = torch.cat([g, corr], dim=-1)

        # 2) 预测 (dx, dy, dtheta)  —— 米+弧度
        raw = self.mlp(g)                                  # [B,3]
        dx_m = self.max_dx * torch.tanh(raw[:, 0:1]) * self.flip_x
        dy_m = self.max_dy * torch.tanh(raw[:, 1:2]) * self.flip_y
        dth  = self.max_rot * torch.tanh(raw[:, 2:3])

        # 3) 转为 grid_sample 的归一化仿射
        #    先把 (dx_m, dy_m) 换成像素，再换成 [-1,1] 归一化平移
        dx_px = dx_m / self.mx    # [B,1]
        dy_px = dy_m / self.my
        tx = 2.0 * dx_px / max(W-1, 1)   # x: 左(-1)→右(+1)
        ty = 2.0 * dy_px / max(H-1, 1)   # y: 上(-1)→下(+1)

        cos_t = torch.cos(dth); sin_t = torch.sin(dth)
        # [[cos, -sin, tx],
        #  [sin,  cos, ty]]
        theta = torch.zeros(B, 2, 3, device=prior_map.device, dtype=prior_map.dtype)
        theta[:, 0, 0] =  cos_t[:, 0]
        theta[:, 0, 1] = -sin_t[:, 0]
        theta[:, 1, 0] =  sin_t[:, 0]
        theta[:, 1, 1] =  cos_t[:, 0]
        theta[:, 0, 2] =  tx[:, 0]
        theta[:, 1, 2] =  ty[:, 0]

        grid = F.affine_grid(theta, size=prior_map.size(), align_corners=align_corners)  # [B,H,W,2]
        prior_warp = F.grid_sample(prior_map, grid, mode=mode, padding_mode='zeros',
                                   align_corners=align_corners)  # [B,C,H,W]

        T_params = {
            'dx_m': dx_m, 'dy_m': dy_m, 'dtheta': dth,
            'dx_norm': tx, 'dy_norm': ty, 'theta_mat': theta
        }
        return prior_warp, T_params

# se2_losses.py
import torch
import torch.nn.functional as F

def feature_align_loss(bev_map, prior_warp, mask=None, topk_ratio=0.2, mode='cos'):
    """
    bev_map, prior_warp: [B,C,H,W]
    mask: [B,1,H,W] 可选（比如 decoder 的热点/置信区域）
    mode: 'cos' or 'l2'
    """
    B, C, H, W = bev_map.shape
    # LayerNorm along C
    def ln(x):
        mu = x.mean(dim=1, keepdim=True); sigma = x.std(dim=1, keepdim=True)+1e-6
        return (x - mu) / sigma
    a = ln(bev_map); p = ln(prior_warp)

    score = (bev_map**2).sum(dim=1, keepdim=True)  # [B,1,H,W] 作为 saliency
    if mask is not None:
        score = score * mask

    score_flat = score.flatten(2)                  # [B,1,HW]
    k = max(1, int(topk_ratio * H * W))
    idx = score_flat.topk(k, dim=-1).indices      # [B,1,K]

    a_flat = a.flatten(2).gather(2, idx.expand(-1, C, -1))  # [B,C,K]
    p_flat = p.flatten(2).gather(2, idx.expand(-1, C, -1))  # [B,C,K]

    if mode == 'cos':
        # 1 - cos
        sim = F.cosine_similarity(a_flat, p_flat, dim=1)  # [B,K]
        loss = (1.0 - sim).mean()
    else:
        loss = F.mse_loss(a_flat, p_flat)

    return loss

# raster_with_se2_and_fuse.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from se2_aligner import SE2Aligner
from se2_losses import feature_align_loss, se2_regularizer

class GatedResidualFuse(nn.Module):
    def __init__(self, c=256, cond_dim=32):
        super().__init__()
        self.ln = nn.LayerNorm(c)
        self.proj = nn.Linear(2*c, c)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.gate = nn.Sequential(nn.Linear(cond_dim, 64), nn.ReLU(True), nn.Linear(64, 2))
        nn.init.constant_(self.gate[-1].bias, -2.0)  # 初始少用先验
    def forward(self, bev_tok, sat_tok=None, sd_tok=None, cond=None):
        B, HW, C = bev_tok.shape
        if sat_tok is None: sat_tok = torch.zeros_like(bev_tok)
        if sd_tok  is None: sd_tok  = torch.zeros_like(bev_tok)
        if cond is None: cond = torch.zeros(B, 32, device=bev_tok.device)
        g = torch.softmax(self.gate(cond), dim=-1)  # [B,2]
        # presence 归一
        pres = torch.stack([sat_tok.abs().sum((-1,-2))>0, sd_tok.abs().sum((-1,-2))>0], dim=-1).float()
        g = g * pres; g = g / g.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        prior = g[:,0].view(B,1,1)*sat_tok + g[:,1].view(B,1,1)*sd_tok
        x = torch.cat([self.ln(bev_tok), self.ln(prior)], dim=-1)
        return bev_tok + self.alpha * self.proj(x)

class RasterPriorWithSE2(nn.Module):
    """
    把 raster 先验(卫星/栅格SD)对齐并融合到 BEV。
    你可以把 sat/sd 的 encoder 换成你现有的 ResNet18 产出 prior_map。
    """
    def __init__(self, H=200, W=100, c=256, m_per_cell=(0.3,0.3),
                 cond_dim=32, max_trans_m=(1.0,1.0), max_rot_deg=5.0,
                 flip_x=False, flip_y=False, align_mode='bilinear'):
        super().__init__()
        self.H, self.W, self.C = H, W, c
        # 这里假设你已有 source-aware ResNet18 得到 sat_map / sd_map
        # 如果需要，这里也可内置 encoder；为简洁起见，本类专注 SE(2)+融合
        self.se2_sat = SE2Aligner(c, H, W, m_per_cell, max_trans_m, max_rot_deg,
                                  cond_dim=cond_dim, use_corr=True, flip_x=flip_x, flip_y=flip_y)
        self.se2_sd  = SE2Aligner(c, H, W, m_per_cell, max_trans_m, max_rot_deg,
                                  cond_dim=cond_dim, use_corr=True, flip_x=flip_x, flip_y=flip_y)
        self.align_mode = align_mode
        self.fuser = GatedResidualFuse(c=c, cond_dim=cond_dim)

    @staticmethod
    def to_tokens(feat_map):  # [B,C,H,W] -> [B,HW,C]
        return feat_map.flatten(2).transpose(1,2).contiguous()

    def forward(self, bev_tok, bev_map,
                sat_map=None, sd_map=None, cond=None,
                return_losses=False, align_weight=0.1, reg_w=0.01):
        """
        bev_tok: [B,HW,C]       bev_map: [B,C,H,W]
        sat_map/sd_map: [B,C,H,W]  (可为 None)
        cond: [B,cond_dim] (如 source embedding/场景id等)
        """
        B = bev_tok.size(0)
        sat_tok = sd_tok = None
        losses = {}

        if sat_map is not None:
            sat_warp, T_sat = self.se2_sat(bev_map, sat_map, cond=cond, mode=self.align_mode)
            sat_tok = self.to_tokens(sat_warp)
            if return_losses:
                L_align_sat = feature_align_loss(bev_map, sat_warp, mask=None, topk_ratio=0.2, mode='cos')
                L_reg_sat   = se2_regularizer(T_sat, w_t=1.0, w_r=1.0)
                losses['L_align_sat'] = L_align_sat * align_weight
                losses['L_reg_sat']   = L_reg_sat   * reg_w

        if sd_map is not None:
            sd_warp, T_sd = self.se2_sd(bev_map, sd_map, cond=cond, mode=self.align_mode)
            sd_tok = self.to_tokens(sd_warp)
            if return_losses:
                L_align_sd = feature_align_loss(bev_map, sd_warp, mask=None, topk_ratio=0.2, mode='cos')
                L_reg_sd   = se2_regularizer(T_sd, w_t=1.0, w_r=1.0)
                losses['L_align_sd'] = L_align_sd * align_weight
                losses['L_reg_sd']   = L_reg_sd   * reg_w

        bev_fused = self.fuser(bev_tok, sat_tok, sd_tok, cond=cond)
        return bev_fused, losses
