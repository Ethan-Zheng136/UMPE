# projects/mmdet3d_plugin/maptr/losses/se2_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES

@LOSSES.register_module()
class SE_Loss(nn.Module):
    def __init__(self, 
                 align_weight=1.0,
                 reg_weight=0.1, 
                 topk_ratio=0.2,
                 align_mode='cos'):
        super().__init__()
        self.align_weight = align_weight
        self.reg_weight = reg_weight
        self.topk_ratio = topk_ratio
        self.align_mode = align_mode
        
    def feature_align_loss(self, bev_map, prior_warp, mask=None):
        B, C, H, W = bev_map.shape
        
        def layer_norm(x):
            mu = x.mean(dim=1, keepdim=True)
            sigma = x.std(dim=1, keepdim=True) + 1e-6
            return (x - mu) / sigma
            
        bev_norm = layer_norm(bev_map)
        prior_norm = layer_norm(prior_warp)
        
        score = (bev_map**2).sum(dim=1, keepdim=True)
        if mask is not None:
            score = score * mask
            
        score_flat = score.flatten(2)
        k = max(1, int(self.topk_ratio * H * W))
        topk_indices = score_flat.topk(k, dim=-1).indices
        
        bev_topk = bev_norm.flatten(2).gather(2, topk_indices.expand(-1, C, -1))
        prior_topk = prior_norm.flatten(2).gather(2, topk_indices.expand(-1, C, -1))
        
        if self.align_mode == 'cos':
            sim = F.cosine_similarity(bev_topk, prior_topk, dim=1)
            loss = (1.0 - sim).mean()
        else:
            loss = F.mse_loss(bev_topk, prior_topk)
            
        return loss
    
    def se2_regularizer(self, T_params):
        dx = T_params['dx_m']
        dy = T_params['dy_m'] 
        dtheta = T_params['dtheta']
        
        trans_reg = (dx**2 + dy**2).mean()
        rot_reg = (dtheta**2).mean()
        
        return trans_reg + rot_reg
    
    def forward(self, bev_map, prior_warp, T_params, **kwargs):
        align_loss = self.feature_align_loss(bev_map, prior_warp)
        reg_loss = self.se2_regularizer(T_params)
        
        total_loss = self.align_weight * align_loss + self.reg_weight * reg_loss
        
        return total_loss