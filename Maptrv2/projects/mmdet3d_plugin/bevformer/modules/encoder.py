
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from projects.mmdet3d_plugin.models.utils.bricks import run_time
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
# from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer, MyCustomBaseTransformerLayer2
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
# from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.cnn.bricks.transformer import TransformerLayerSequence, POSITIONAL_ENCODING, build_positional_encoding
from mmcv.cnn import xavier_init, constant_init
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner.base_module import BaseModule
import numpy as np
import torch
import cv2 as cv
import mmcv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
from .spatial_cross_attention import CrossAttentionBEV
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, pc_range=None, num_points_in_pillar=4, return_intermediate=False, dataset_type='nuscenes',
                 **kwargs):

        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range,  img_metas):

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask

    @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = bev_query
        intermediate = []

        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar, dim='3d', bs=bev_query.size(1),  device=bev_query.device, dtype=bev_query.dtype)
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, kwargs['img_metas'])

        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        # shift_ref_2d = ref_2d  # .clone()
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack(
                [prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':

                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayer2(MyCustomBaseTransformerLayer2):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 ffn_cfgs=None,  ## sdmap
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 **kwargs):
        super(BEVFormerLayer2, self).__init__(
            attn_cfgs=attn_cfgs,
            ffn_cfgs=ffn_cfgs,  ## sdmap
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            **kwargs)
        self.fp16_enabled = False

    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                map_bev=None,
                map_graph_feats=None,
                map_graph_shapes=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':

                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            # bev cross attention  (sdmap)
            elif layer == 'cross_attn_bev':
                query = self.attentions[attn_index](
                    query,
                    None,
                    map_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    reference_points=ref_2d,
                    attn_mask=attn_masks[attn_index],
                    spatial_shapes=bev_spatial_shapes,
                    level_start_index=bev_level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query
            
            # graph cross attention
            elif layer == 'cross_attn_graph':
                query = self.attentions[attn_index](
                    query,
                    None,
                    map_graph_feats,
                    residual=query,
                    query_pos=bev_pos,
                    spatial_shapes=map_graph_shapes,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MapGraphTransformer(BaseModule):
    def __init__(self, 
                 input_dim=30,  # 2 * 11 points + 8 classes
                 dmodel=256,
                 hidden_dim=2048,  # set this to something smaller
                 nheads=8,
                 nlayers=6,
                 batch_first=False,  # set to True
                 pos_encoder=None,
                 **kwargs):
        super(MapGraphTransformer, self).__init__(**kwargs)
        self.batch_dim = 0 if batch_first else 1
        self.map_embedding = nn.Linear(input_dim, dmodel)
        

        if pos_encoder is not None:
            self.use_positional_encoding = True
            self.pos_encoder = build_positional_encoding(pos_encoder)
        else:
            self.use_positional_encoding = False
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=dmodel,
                                                        nhead=nheads,
                                                        dim_feedforward=hidden_dim,
                                                        batch_first=batch_first)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.init_weights()

    def init_weights(self):
        xavier_init(self.map_embedding, distribution='uniform')

        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                xavier_init(p, distribution='uniform')

        if hasattr(self, 'pos_encoder') and self.pos_encoder is not None:
            if hasattr(self.pos_encoder, 'init_weights'):
                self.pos_encoder.init_weights()

    def process_map_data(self, prior_map, map_type="sdmap", img_feats=None):
        TYPE_VOCAB = {
            "divider": 0,       # hd_divider
            "ped_crossing": 1,  # hd_ped_crossing
            "boundary": 2,      # hd_boundary
            "highway": 3,       # sd_highway
            "primary": 4,       # sd_primary
            "secondary": 5,     # sd_secondary
            "tertiary": 6,      # sd_tertiary
            "residential": 7,   # sd_residential
            "service": 8,       # sd_service
            "pedestrian": 9,    # sd_pedestrian
            "other": 10,        # sd_other
        }
        n_points = 11
        
        device = img_feats[0].device
        dtype = torch.float32
        
        num_categories = max(TYPE_VOCAB.values()) + 1

        source_code = torch.tensor([1, 0], dtype=dtype, device=device) if map_type == 'sdmap' else torch.tensor([0, 1], dtype=dtype, device=device)

        map_graph_list = [] 
        onehot_category_list = [] 
        onehot_source_list = []   

        for category, polylines in prior_map.items():
            if len(polylines) > 0:
                lane_category_onehot = torch.zeros(num_categories, dtype=dtype, device=device)
                lane_category_onehot[TYPE_VOCAB[category]] = 1.0
                
                for polyline in polylines:
                    polyline = polyline.squeeze(0)
                    map_graph_list.append(polyline)
                    onehot_category_list.append(lane_category_onehot.clone())
                    onehot_source_list.append(source_code.clone())

        if len(map_graph_list) == 0:
            map_graph = torch.zeros((1, n_points, 2), dtype=dtype, device=device)
            onehot_category = torch.zeros((1, num_categories), dtype=dtype, device=device)
            onehot_source = source_code.clone().unsqueeze(0)
        else:
            map_graph = torch.stack(map_graph_list, dim=0) ## npolyline * 11 * 2
            onehot_category = torch.stack(onehot_category_list, dim=0)  ## npolyline * 11
            onehot_source = torch.stack(onehot_source_list, dim=0)  ## npolyline * 2

        return map_graph.unsqueeze(0), onehot_category.unsqueeze(0), onehot_source.unsqueeze(0)

    def forward(self, sdmap, map_type, img_feats):
        # batch_map_graph: list of polylines
        # batch, num_polylines * points * 2 (x, y)
        # onehot_category: batch, num_polylines * num_categories, onehot encoding of categories
        # TODO: make batched
        if map_type is not None:
            map_graph, onehot_category, onehot_source = self.process_map_data(sdmap, map_type, img_feats)
        else:
            map_graph, onehot_category, onehot_source = sdmap
        # map_graph, onehot_category, onehot_source = sdmap['points'], sdmap['type_onehot'], sdmap['source_onehot']
        # print(map_graph.shape, onehot_category.shape)
        # print(map_graph.shape, onehot_category.shape, onehot_source.shape)

        batch_graph_feats = []
        for graph_polylines, onehot_cat, onehot_src in zip(map_graph, onehot_category, onehot_source):
            if self.use_positional_encoding:
                graph_polylines = self.pos_encoder(graph_polylines)  ## npolylines, npoints(11), 32
            
            npolylines, npoints, pdim = graph_polylines.shape  # unfold the batch
            graph_feat = torch.cat([graph_polylines.view(npolylines, npoints * pdim), onehot_cat, onehot_src], dim=-1) ## npolylines, 365

            # embed features
            graph_feat = self.map_embedding(graph_feat)  # num_polylines, dmodel

            graph_feat = self.transformer_encoder(graph_feat.unsqueeze(self.batch_dim))  # 1, num_polylines, hidden_dim

            confidence_token = self.confidence_head(graph_feat).squeeze(-1) # 1, num_polylines

        return graph_feat, confidence_token


@POSITIONAL_ENCODING.register_module()
class SineContinuousPositionalEncoding(BaseModule):
    def __init__(self, 
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 range=None,
                 scale=2 * np.pi,
                 offset=0.,
                 init_cfg=None):
        super(SineContinuousPositionalEncoding, self).__init__(init_cfg)
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.range = torch.tensor(range) if range is not None else None
        self.offset = torch.tensor(offset) if offset is not None else None
        self.scale = scale
        self.init_weights()

    def init_weights(self):
        pass
    
    def forward(self, x):
        """
        x: [B, N, D]

        return: [B, N, D * num_feats]
        """
        B, N, D = x.shape
        if self.normalize:
            x = (x - self.offset.to(x.device)) / self.range.to(x.device) * self.scale
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = x[..., None] / dim_t  # [B, N, D, num_feats]
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=3).view(B, N, D * self.num_feats)
        return pos_x

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class AlignNet(BaseModule):
    def __init__(self, 
                 hidden_dim=256,
                 in_channels=256, 
                 ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)
        )
         
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        init.zeros_(self.mlp[-1].weight)
        init.zeros_(self.mlp[-1].bias)

    def process_map_data(self, prior_map, map_type="sdmap", img_feats=None):
        TYPE_VOCAB = {
            "divider": 0,       # hd_divider
            "ped_crossing": 1,  # hd_ped_crossing
            "boundary": 2,      # hd_boundary
            "highway": 3,       # sd_highway
            "primary": 4,       # sd_primary
            "secondary": 5,     # sd_secondary
            "tertiary": 6,      # sd_tertiary
            "residential": 7,   # sd_residential
            "service": 8,       # sd_service
            "pedestrian": 9,    # sd_pedestrian
            "other": 10,        # sd_other
        }
        n_points = 11

        device = img_feats[0].device
        dtype = torch.float32
        
        num_categories = max(TYPE_VOCAB.values()) + 1
        source_code = torch.tensor([1, 0], dtype=dtype, device=device) if map_type == 'sdmap' else torch.tensor([0, 1], dtype=dtype, device=device)

        map_graph_list = [] 
        onehot_category_list = [] 
        onehot_source_list = []  

        for category, polylines in prior_map.items():
            if len(polylines) > 0:
                lane_category_onehot = torch.zeros(num_categories, dtype=dtype, device=device)
                lane_category_onehot[TYPE_VOCAB[category]] = 1.0

                for polyline in polylines:
                    polyline = polyline.squeeze(0)
                    map_graph_list.append(polyline)
                    onehot_category_list.append(lane_category_onehot.clone())
                    onehot_source_list.append(source_code.clone())

        if len(map_graph_list) == 0:
            map_graph = torch.zeros((1, n_points, 2), dtype=dtype, device=device)
            onehot_category = torch.zeros((1, num_categories), dtype=dtype, device=device)
            onehot_source = source_code.clone().unsqueeze(0)
        else:
            map_graph = torch.stack(map_graph_list, dim=0) ## npolyline * 11 * 2
            onehot_category = torch.stack(onehot_category_list, dim=0)  ## npolyline * 11
            onehot_source = torch.stack(onehot_source_list, dim=0)  ## npolyline * 2

        return [map_graph.unsqueeze(0), onehot_category.unsqueeze(0), onehot_source.unsqueeze(0)]
        
    def forward(self, bev_embed, sd_token, sdmap, map_type, img_feats):
        sdmap = self.process_map_data(sdmap, map_type, img_feats)
        B, npolylines, npoints, _ = sdmap[0].shape
        
        combined = torch.cat([bev_embed, sd_token], dim=1)  # [bs, c, 256]
        combined_pooled = self.pool(combined.transpose(1, 2)).squeeze(-1)
        params = self.mlp(combined_pooled)  # [B, 3]
        
        theta_raw = torch.sigmoid(params[:, 0])  # [B]
        theta = (theta_raw * 2 - 1) * math.pi  # [-π, π]
        
        t = params[:, 1:3]  # [B, 2]

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        R = torch.zeros(B, 2, 2, device=sdmap[0].device)
        R[:, 0, 0] = sin_theta
        R[:, 0, 1] = cos_theta
        R[:, 1, 0] = cos_theta
        R[:, 1, 1] = -sin_theta
        
        sdmap_flat = sdmap[0].reshape(B, -1, 2)  # [B, npolylines * 11, 2]
        
        sdmap_flat_t = sdmap_flat.transpose(1, 2)  # [B, 2, npolylines * 11]

        t_expanded = t.unsqueeze(2)  # [B, 2, 1]

        sd_aligned_flat_t = torch.bmm(R, sdmap_flat_t) + t_expanded  # [B, 2, npolylines * 11]
        
        sd_aligned_flat = sd_aligned_flat_t.transpose(1, 2)  # [B, npolylines * 11, 2]

        sdmap_aligned = sd_aligned_flat.reshape(B, npolylines, npoints, 2)
        sdmap[0] = sdmap_aligned

        transform_params = {
            'theta': theta,
            't': t,
            'R': R
        }
        
        return sdmap, transform_params


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DualVecCrossFusion(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 source_dropout_p=0.3,
                 bev_w=100,
                 bev_h=200,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.bev_w = bev_w
        self.bev_h = bev_h
        self.ca_hd = CrossAttentionBEV(embed_dims=embed_dims, num_heads=num_heads)
        self.ca_sd = CrossAttentionBEV(embed_dims=embed_dims, num_heads=num_heads)

        self.gate_logits = nn.Parameter(torch.tensor([0.0, 0.0]))  # [HD, SD]

        self.source_dropout_p = source_dropout_p

    def init_weights(self):
        super().init_weights()
        nn.init.constant_(self.gate_logits, 0.0)

    def _maybe_source_dropout(self, has_hd, has_sd):
        use_hd, use_sd = has_hd, has_sd
        if self.training and has_hd and has_sd and self.source_dropout_p > 0:
            if torch.rand(1).item() < self.source_dropout_p:
                if torch.rand(1).item() < 0.5:
                    use_hd = False
                else:
                    use_sd = False
        return use_hd, use_sd

    def forward(self,
                bev_tok,                     # [B,Nq,C]
                hd_tokens=None, hd_U=None,   # [B,Nh,C], [B,Nh]
                sd_tokens=None, sd_U=None):  # [B,Ns,C], [B,Ns]
        B, Nq, C = bev_tok.shape

        has_hd = (hd_tokens is not None) and (hd_tokens.size(1) > 0)
        has_sd = (sd_tokens is not None) and (sd_tokens.size(1) > 0)
        use_hd, use_sd = self._maybe_source_dropout(has_hd, has_sd)

        y_hd = torch.zeros_like(bev_tok)
        y_sd = torch.zeros_like(bev_tok)

        if use_hd:
            y_hd = self.ca_hd(bev_tok, hd_tokens, hd_U)  # [B,Nq,C]
        if use_sd:
            y_sd = self.ca_sd(bev_tok, sd_tokens, sd_U)  # [B,Nq,C]

        pres = torch.tensor([float(use_hd), float(use_sd)], device=bev_tok.device)  # [2]

        g = torch.softmax(self.gate_logits, dim=0) * pres
        g = g / g.sum().clamp_min(1e-6)

        fused = g[0] * y_hd + g[1] * y_sd             # [B,Nq,C]

        return fused, {'gate': g.detach(), 'use_hd': use_hd, 'use_sd': use_sd}
