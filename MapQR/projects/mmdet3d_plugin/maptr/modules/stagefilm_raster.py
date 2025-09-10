import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models import resnet50
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
import math


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFusion(nn.Module):
    def __init__(self, embed_dims, bev_h=200, bev_w=100):
        super().__init__()
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        
        self.safe_fuse = SafeResidualFuse(c=embed_dims, init_alpha=0.0, freeze_alpha=False,
                                         bev_h=bev_h, bev_w=bev_w)
            
    def forward(self, bev_embed, prior_bev):
        # print('BEVFusion')
        fused_bev = self.safe_fuse(bev_embed, prior_bev)

        return fused_bev

class SafeResidualFuse(nn.Module):
    def __init__(self, bev_h=200, bev_w=100, c=256, init_alpha=0.0, freeze_alpha=False):
        super().__init__()
        self.ln = nn.LayerNorm(c)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2 * c, c, 1), 
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True)
        )
        self.alpha = nn.Parameter(torch.tensor(init_alpha)) 

        if freeze_alpha:
            self._freeze_alpha()
        
    def _freeze_alpha(self):
        self.alpha.requires_grad = False 
        
    def forward(self, bev_embed, prior_bev):
        B = bev_embed.shape[0]
        # layernorm
        # print('-----yes-----')
        bev_embed = self.ln(bev_embed)
        prior_bev = self.ln(prior_bev)
        # print('self.alpha:', self.alpha.device)
        concat_features = torch.cat([bev_embed, prior_bev], dim=-1)  # [1, 20000, 512]

        concat_2d = concat_features.view(B, self.bev_h, self.bev_w, -1).permute(0, 3, 1, 2).contiguous()  # [1, 512, 200, 100]
        fused_2d = self.fusion_conv(concat_2d)  # [1, 256, 200, 100]
        fused_bev = fused_2d.flatten(2).transpose(1, 2)  # [1, 20000, 256]
    
        return bev_embed + self.alpha * fused_bev


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class StageFiLMRasterEncoder(nn.Module):
    def __init__(self, embed_dims=256, bev_h=200, bev_w=100, cond_dim=64, 
                 pretrained=True, freeze_layers=True, num_groups=16):
        super().__init__()
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        
        backbone = resnet18(pretrained=pretrained)
        
        self.conv1 = backbone.conv1     
        self.bn1 = backbone.bn1          
        self.relu = backbone.relu       
        self.maxpool = backbone.maxpool  
        
        self.stage1 = backbone.layer1 
        self.stage2 = backbone.layer2 
        self.stage3 = backbone.layer3  
        self.stage4 = backbone.layer4 
        
        self.film1 = FiLM(64, cond_dim) 
        self.film2 = FiLM(128, cond_dim) 
        self.film3 = FiLM(256, cond_dim)
        self.film4 = FiLM(512, cond_dim)
        
        self.src_embed = nn.Embedding(2, cond_dim)  # [sat, sd]
        self.gate = BEVGate(feature_dim=embed_dims, bev_h=bev_h, bev_w=bev_w)
        # self.gate = CondGate(embed_dims=embed_dims, bev_h=bev_h, bev_w=bev_w, cond_dim=cond_dim)
        self.source_dropout = SourceDropout(p_sat=0.2, p_sd=0.2)

        if freeze_layers:
            self._freeze_layers()
        
        self.final_conv = nn.Conv2d(512, embed_dims, 1)  # 512 → 256
        self.final_gn = nn.GroupNorm(num_groups=num_groups, num_channels=embed_dims)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((bev_h, bev_w))
        
    def _freeze_layers(self):
        stages_to_freeze = [self.conv1, self.bn1, self.stage1, self.stage2]
        for stage in stages_to_freeze:
            for param in stage.parameters():
                param.requires_grad = False
            stage.eval()
            
    def _build_cond(self, sat, sd, B, device):
        cond = torch.zeros(B, self.src_embed.embedding_dim, device=device)
        
        if sat is not None:
            cond += self.src_embed.weight[0].unsqueeze(0) 
        if sd is not None:
            cond += self.src_embed.weight[1].unsqueeze(0) 
            
        return cond

    def _forward_with_film(self, x, cond):
        x = self.conv1(x) 
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) 
        x = self.stage1(x) 
        x = self.film1(x, cond)
        x = self.stage2(x)
        x = self.film2(x, cond)
        x = self.stage3(x)
        x = self.film3(x, cond) 
        x = self.stage4(x) 
        x = self.film4(x, cond)
        return x
    
    def forward(self, sat=None, sd=None):
        sat, sd = self.source_dropout(sat, sd)
        if sat is None and sd is None:
            return None
        
        results = {}
        B = None
        device = None
        if sat is not None:
            B, H, W, C = sat.shape
            device = sat.device
        if sd is not None:
            B, H, W, C = sd.shape
            device = sd.device
        cond = self._build_cond(sat, sd, B, device)

        if sat is not None:
            B, H, W, C = sat.shape
            device = sat.device
            sat = sat.permute(0, 3, 1, 2)  # [1, 3, 200, 100]

            sat_feat = self._forward_with_film(sat, cond)    # [1, 512, 6, 3]

            sat_feat = self.final_conv(sat_feat)             # [1, 256, 6, 3]
            sat_feat = self.final_gn(sat_feat)
            sat_feat = self.adaptive_pool(sat_feat)          # [1, 256, 200, 100]
            
            results['sat_feat'] = sat_feat
            # print(f"sat_prior: {sat_prior.shape}")
        
        if sd is not None:
            if B is None:
                B, H, W, C = sd.shape
                device = sd.device
            sd = sd.permute(0, 3, 1, 2)                     # [1, 3, 200, 100]     

            sd_feat = self._forward_with_film(sd, cond)     # [1, 512, 6, 3]
            
            sd_feat = self.final_conv(sd_feat)              # [1, 256, 6, 3]
            sd_feat = self.final_gn(sd_feat)
            sd_feat = self.adaptive_pool(sd_feat)           # [1, 256, 200, 100]
            sd_prior = sd_feat.flatten(2).transpose(1, 2)   # [1, 20000, 256]
            
            results['sd_feat'] = sd_feat
            # print(f"sd_prior: {sd_prior.shape}")
        
        if 'sat_feat' in results and 'sd_feat' in results:
            # BEVgate
            prior_bev = self.gate(results['sat_feat'], results['sd_feat'])
            # presence gate
            # prior_bev = self.gate(results['sat_feat'], results['sd_feat'], cond)
        elif 'sat_feat' in results and 'sd_feat' not in results:
            sat_prior = results['sat_feat'].flatten(2).transpose(1, 2)
            prior_bev = sat_prior
        elif 'sat_feat' not in results and 'sd_feat' in results:
            sd_prior = results['sd_feat'].flatten(2).transpose(1, 2)
            prior_bev = sd_prior
        else:
            prior_bev = None
            
        return prior_bev



class FiLM(nn.Module):
    def __init__(self, c: int, d: int):
        super().__init__()
        self.fc = nn.Linear(d, 2*c)
        
    def forward(self, x, cond):
        gb = self.fc(cond).unsqueeze(-1).unsqueeze(-1)  # [B, 2C, 1, 1]
        C = x.size(1)
        gamma, beta = gb[:, :C], gb[:, C:]              # [B, C, 1, 1] each
        return x * (1.0 + gamma) + beta               


class SourceDropout(nn.Module):
    def __init__(self, p_sat=0.2, p_sd=0.2):
        super().__init__()
        self.p_sat, self.p_sd = p_sat, p_sd
    def forward(self, sat=None, sd=None):
        if not self.training: return sat, sd
        if sat is not None and torch.rand(1).item() < self.p_sat: sat = None
        if sd  is not None and torch.rand(1).item() < self.p_sd:  sd  = None
        if sat is None and sd is None:
            if torch.rand(1).item() < 0.5: sat = torch.zeros_like(sd) if sd is not None else None
            else: sd = torch.zeros_like(sat) if sat is not None else None
        return sat, sd


def convert_bn_to_gn(module, num_groups=16):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            gn = nn.GroupNorm(num_groups=num_groups, num_channels=child.num_features)
            setattr(module, name, gn)
        else:
            convert_bn_to_gn(child, num_groups)


class BEVGate(nn.Module):
    def __init__(self, feature_dim=256, bev_h=200, bev_w=100):
        super().__init__()
        self.bev_h, self.bev_w = bev_h, bev_w
        self.feature_dim = feature_dim

        self.gate_conv = nn.Sequential(
            nn.Conv2d(feature_dim * 2, 128, 3, padding=1), 
            nn.GroupNorm(8, 128), 
            nn.ReLU(),
            
            nn.Conv2d(128, 64, 3, padding=1),  
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d(1),                         # [B, 64, 1, 1]
            nn.Conv2d(64, feature_dim * 2, 1),               # 64 -> 512 (256*2)
        )
        
    def forward(self, sat_bev, sd_bev):
        B, C, H, W = sat_bev.shape                           # [B, 256, 200, 100]

        concat_bev = torch.cat([sat_bev, sd_bev], dim=1)     # [B, 512, 200, 100]
        
        weights = self.gate_conv(concat_bev)                 # [B, 512, 1, 1]
        weights = weights.squeeze(-1).squeeze(-1)            # [B, 512]
        dual_weights = weights.view(B, C, 2)                 # [B, 512] -> [B, 256, 2]
        normalized_weights = F.softmax(dual_weights, dim=2)  # [B, 256, 2]
        
        sat_weights = normalized_weights[:, :, 0]            # [B, 256]
        sd_weights = normalized_weights[:, :, 1]             # [B, 256]
        
        sat_weights = sat_weights.view(B, C, 1, 1)           # [B, 256, 1, 1] 广播用
        sd_weights = sd_weights.view(B, C, 1, 1)             # [B, 256, 1, 1] 广播用
        
        fused_bev = sat_weights * sat_bev + sd_weights * sd_bev  # [B, 256, 200, 100]
        
        fused_prior = fused_bev.view(B, -1, self.bev_h * self.bev_w).transpose(1, 2)  # [B, 20000, 256]
        
        return fused_prior



class SE2Aligner(nn.Module):
    def __init__(self,
                 c=256, H=200, W=100,
                 m_per_cell=(0.3, 0.3), 
                 max_trans_m=(1.0, 1.0),  
                 max_rot_deg=10.0,            
                 cond_dim=32,                  
                 use_corr=False,                 
                 flip_x=False, flip_y=False):   
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
            self.corr_pool = nn.AdaptiveAvgPool2d((1,1))
            in_dim += c

        self.head = nn.Sequential(
            nn.Conv2d(2*c, c, 1, bias=False),   
            nn.GroupNorm(16, c), nn.ReLU(True)
        )
        mlp_in = c + (cond_dim if cond_dim>0 else 0) + (c if use_corr else 0)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 128), nn.ReLU(True),
            nn.Linear(128, 3)
        )
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

        x = torch.cat([bev_map, prior_map], dim=1)         # [B,2C,H,W]
        x = self.head(x)                                   # [B,C,H,W]
        g = self.gap(x).flatten(1)                         # [B,C]
        if cond is not None:
            g = torch.cat([g, cond], dim=-1)               # [B,C+D]
        if self.use_corr:
            corr = self.corr_pool(bev_map * prior_map).flatten(1)  # [B,C]
            g = torch.cat([g, corr], dim=-1)

        raw = self.mlp(g)                                  # [B,3]
        dx_m = self.max_dx * torch.tanh(raw[:, 0:1]) * self.flip_x
        dy_m = self.max_dy * torch.tanh(raw[:, 1:2]) * self.flip_y
        dth  = self.max_rot * torch.tanh(raw[:, 2:3])

        dx_px = dx_m / self.mx    # [B,1]
        dy_px = dy_m / self.my
        tx = 2.0 * dx_px / max(W-1, 1)  
        ty = 2.0 * dy_px / max(H-1, 1)  

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
