# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm1D(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        hidden_dim = heads * dim_head
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            nn.GroupNorm(1, dim),
        )

    def forward(self, x):
        b, _, n = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q = q.reshape(b, self.heads, self.dim_head, n)
        k = k.reshape(b, self.heads, self.dim_head, n)
        v = v.reshape(b, self.heads, self.dim_head, n)

        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = out.reshape(b, self.heads * self.dim_head, n)
        return self.to_out(out)


class StandardAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        hidden_dim = heads * dim_head
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, _, n = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q = q.reshape(b, self.heads, self.dim_head, n) * self.scale
        k = k.reshape(b, self.heads, self.dim_head, n)
        v = v.reshape(b, self.heads, self.dim_head, n)

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = out.permute(0, 1, 3, 2).reshape(b, self.heads * self.dim_head, n)
        return self.to_out(out)


class Initial3DConv(nn.Module):
    def __init__(self, in_channels=6, out_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 7), padding=(1, 1, 3)),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)


class SpatioTemporalFusion3D(nn.Module):
    def __init__(self, in_channels=64, out_channels=128):
        super().__init__()
        mid_channels = (in_channels + out_channels) // 2

        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=(3, 3, 7), padding=(1, 1, 3)),
            nn.GroupNorm(_group_count(mid_channels), mid_channels),
            nn.SiLU(),
        )
        self.down1 = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, kernel_size=(2, 2, 3), stride=(2, 2, 1), padding=(0, 0, 1)),
            nn.GroupNorm(_group_count(mid_channels), mid_channels),
            nn.SiLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv3d(mid_channels, out_channels, kernel_size=(2, 2, 5), stride=(2, 2, 1), padding=(0, 0, 2)),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, 5), padding=(0, 0, 2)),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
        )
        self.up1 = nn.ConvTranspose3d(out_channels, mid_channels, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.dec1 = nn.Sequential(
            nn.Conv3d(mid_channels * 2, mid_channels, kernel_size=(3, 3, 5), padding=(1, 1, 2)),
            nn.GroupNorm(_group_count(mid_channels), mid_channels),
            nn.SiLU(),
        )
        self.up2 = nn.ConvTranspose3d(mid_channels, mid_channels, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.dec2 = nn.Sequential(
            nn.Conv3d(mid_channels * 2, mid_channels, kernel_size=(3, 3, 5), padding=(1, 1, 2)),
            nn.GroupNorm(_group_count(mid_channels), mid_channels),
            nn.SiLU(),
        )
        self.out_conv = nn.Sequential(
            nn.Conv3d(mid_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)

        h = self.bottleneck(e3)
        h = self.up1(h)
        h = self.dec1(torch.cat([h, e2], dim=1))
        h = self.up2(h)
        h = self.dec2(torch.cat([h, e1], dim=1))
        return self.out_conv(h)


class SpatialPooling(nn.Module):
    def __init__(self, in_channels=128, roi_n=4):
        super().__init__()
        num_pools = int(math.log2(roi_n))
        if roi_n < 2 or 2 ** num_pools != roi_n:
            raise ValueError(f"roi_n must be a power of two, got {roi_n}")

        self.pools = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=(2, 2, 1), stride=(2, 2, 1)),
                nn.GroupNorm(_group_count(in_channels), in_channels),
                nn.SiLU(),
            )
            for _ in range(num_pools)
        ])

    def forward(self, x):
        for pool in self.pools:
            x = pool(x)
        return x


class ConditionProcessor(nn.Module):
    def __init__(self, c_channels=3, out_dim=128):
        super().__init__()
        self.c_channels = c_channels
        mid_dim = out_dim // 2

        self.cond_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, mid_dim // 2, kernel_size=7, padding=3),
                nn.SiLU(),
                nn.Conv1d(mid_dim // 2, mid_dim, kernel_size=3, padding=1),
            )
            for _ in range(c_channels)
        ])
        self.cond_weight_net = nn.Sequential(
            nn.Conv1d(mid_dim * c_channels, mid_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(mid_dim, c_channels, kernel_size=1),
            nn.Softmax(dim=1),
        )
        self.proj = nn.Sequential(
            nn.Conv1d(mid_dim, out_dim, kernel_size=1),
            nn.GroupNorm(1, out_dim),
            nn.SiLU(),
        )

    def forward(self, c_raw):
        _, channels, _ = c_raw.shape
        if channels != self.c_channels:
            raise ValueError(f"Expected {self.c_channels} traditional signal channels, got {channels}")

        cond_features = []
        for i, conv in enumerate(self.cond_convs):
            cond_features.append(conv(c_raw[:, i:i + 1, :]))

        cond_stack = torch.stack(cond_features, dim=1)
        cond_weights = self.cond_weight_net(cond_stack.flatten(1, 2)).unsqueeze(2)
        return self.proj((cond_stack * cond_weights).sum(dim=1))


class AdaLN1D(nn.Module):
    def __init__(self, num_features, time_emb_dim):
        super().__init__()
        self.norm = nn.GroupNorm(1, num_features, affine=False)
        self.ada_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, num_features * 2),
        )

    def forward(self, x, time_emb):
        gamma, beta = self.ada_proj(time_emb).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return self.norm(x) * (1 + gamma) + beta


class AdaLNResBlock(nn.Module):
    def __init__(self, channels, time_emb_dim, dropout=0.0):
        super().__init__()
        self.adaln1 = AdaLN1D(channels, time_emb_dim)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.adaln2 = AdaLN1D(channels, time_emb_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x, time_emb):
        h = self.conv1(F.silu(self.adaln1(x, time_emb)))
        h = self.conv2(self.dropout(F.silu(self.adaln2(h, time_emb))))
        return x + h


class Downsample1D(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv1d(dim, dim_out, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class ConvSimpleVelocityPredictor(nn.Module):
    def __init__(self, input_channels=1, feature_dim=128, time_emb_dim=256, hidden_dim=64):
        super().__init__()
        self.input_channels = input_channels
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, feature_dim),
            nn.SiLU(),
        )
        self.refine = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.SiLU(),
        )
        self.predict = nn.Sequential(
            nn.Conv1d(hidden_dim + input_channels, hidden_dim, kernel_size=7, padding=3),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(hidden_dim // 2), hidden_dim // 2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
        )
        nn.init.zeros_(self.predict[-1].weight)
        nn.init.zeros_(self.predict[-1].bias)

    def forward(self, fused_cond, x_t, time_emb):
        if fused_cond.shape[-1] != x_t.shape[-1]:
            fused_cond = F.interpolate(fused_cond, size=x_t.shape[-1], mode="linear", align_corners=False)

        feat = fused_cond + self.time_proj(time_emb).unsqueeze(-1)
        feat = self.refine(feat)
        return self.predict(torch.cat([feat, x_t], dim=1))


class GatedFusionWithRefinement(nn.Module):
    def __init__(self, time_emb_dim=256, hidden_dim=32):
        super().__init__()
        self.gate_conv1 = nn.Conv1d(2, hidden_dim, kernel_size=3, padding=1)
        self.gate_norm = nn.GroupNorm(_group_count(hidden_dim, 4), hidden_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)
        self.gate_conv2 = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.refine = nn.Sequential(
            nn.Conv1d(4, hidden_dim, kernel_size=5, padding=2),
            nn.GroupNorm(_group_count(hidden_dim, 4), hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(hidden_dim, 4), hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, v_stmap, v_c, time_emb):
        gate_feat = self.gate_conv1(torch.cat([v_stmap, v_c], dim=1))
        gate_feat = F.silu(self.gate_norm(gate_feat))
        gate_feat = gate_feat + self.time_proj(time_emb).unsqueeze(-1)
        gate = torch.sigmoid(self.gate_conv2(gate_feat))

        blended = gate * v_stmap + (1 - gate) * v_c
        residual = self.refine(torch.cat([blended, v_stmap, v_c, gate], dim=1))
        return blended + residual, gate, residual


class AdaGNUNetVelocityPredictorV2(nn.Module):
    def __init__(
        self,
        input_channels=1,
        feature_dim=128,
        time_emb_dim=256,
        base_dim=64,
        num_down_layers=2,
        dim_mults=(1, 2, 4),
        dropout=0.0,
        attn_heads=4,
        attn_dim_head=32,
    ):
        super().__init__()

        if len(dim_mults) < num_down_layers + 1:
            dim_mults = tuple(dim_mults) + (dim_mults[-1],) * (num_down_layers + 1 - len(dim_mults))
        dim_mults = tuple(dim_mults[:num_down_layers + 1])
        dims = [base_dim * int(mult) for mult in dim_mults]

        self.num_down_layers = num_down_layers
        self.input_proj = nn.Conv1d(input_channels + feature_dim, dims[0], kernel_size=3, padding=1)

        self.encoder_blocks1 = nn.ModuleList()
        self.encoder_blocks2 = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(num_down_layers):
            dim_in = dims[i]
            dim_out = dims[i + 1]
            self.encoder_blocks1.append(AdaLNResBlock(dim_in, time_emb_dim, dropout=dropout))
            self.encoder_blocks2.append(AdaLNResBlock(dim_in, time_emb_dim, dropout=dropout))
            self.encoder_attns.append(
                Residual(PreNorm1D(dim_in, LinearAttention(dim_in, heads=attn_heads, dim_head=attn_dim_head)))
            )
            self.downsamples.append(Downsample1D(dim_in, dim_out))

        bottleneck_dim = dims[-1]
        self.bottleneck_block1 = AdaLNResBlock(bottleneck_dim, time_emb_dim, dropout=dropout)
        self.bottleneck_attn = Residual(
            PreNorm1D(bottleneck_dim, StandardAttention(bottleneck_dim, heads=attn_heads, dim_head=attn_dim_head))
        )
        self.bottleneck_block2 = AdaLNResBlock(bottleneck_dim, time_emb_dim, dropout=dropout)

        self.upsamples = nn.ModuleList()
        self.decoder_merge1 = nn.ModuleList()
        self.decoder_blocks1 = nn.ModuleList()
        self.decoder_merge2 = nn.ModuleList()
        self.decoder_blocks2 = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()

        for i in range(num_down_layers - 1, -1, -1):
            dim_in = dims[i + 1]
            dim_out = dims[i]
            self.upsamples.append(Upsample1D(dim_in, dim_out))
            self.decoder_merge1.append(nn.Conv1d(dim_out * 2, dim_out, kernel_size=1))
            self.decoder_blocks1.append(AdaLNResBlock(dim_out, time_emb_dim, dropout=dropout))
            self.decoder_merge2.append(nn.Conv1d(dim_out * 2, dim_out, kernel_size=1))
            self.decoder_blocks2.append(AdaLNResBlock(dim_out, time_emb_dim, dropout=dropout))
            self.decoder_attns.append(
                Residual(PreNorm1D(dim_out, LinearAttention(dim_out, heads=attn_heads, dim_head=attn_dim_head)))
            )

        self.final_norm = nn.GroupNorm(1, dims[0])
        self.final_act = nn.SiLU()
        self.out_conv = nn.Conv1d(dims[0], 1, kernel_size=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, fused_cond, x_t, time_emb):
        original_t = x_t.shape[-1]
        if fused_cond.shape[-1] != original_t:
            fused_cond = F.interpolate(fused_cond, size=original_t, mode="linear", align_corners=False)

        h = self.input_proj(torch.cat([x_t, fused_cond], dim=1))
        skips = []

        for i in range(self.num_down_layers):
            h = self.encoder_blocks1[i](h, time_emb)
            skips.append(h)
            h = self.encoder_blocks2[i](h, time_emb)
            skips.append(h)
            h = self.encoder_attns[i](h)
            h = self.downsamples[i](h)

        h = self.bottleneck_block1(h, time_emb)
        h = self.bottleneck_attn(h)
        h = self.bottleneck_block2(h, time_emb)

        for i in range(self.num_down_layers):
            enc_layer_idx = self.num_down_layers - 1 - i
            skip1 = skips[enc_layer_idx * 2]
            skip2 = skips[enc_layer_idx * 2 + 1]

            h = self.upsamples[i](h)
            if h.shape[-1] != skip2.shape[-1]:
                h = F.interpolate(h, size=skip2.shape[-1], mode="linear", align_corners=False)

            h = self.decoder_merge1[i](torch.cat([h, skip2], dim=1))
            h = self.decoder_blocks1[i](h, time_emb)
            h = self.decoder_merge2[i](torch.cat([h, skip1], dim=1))
            h = self.decoder_blocks2[i](h, time_emb)
            h = self.decoder_attns[i](h)

        if h.shape[-1] != original_t:
            h = F.interpolate(h, size=original_t, mode="linear", align_corners=False)

        return self.out_conv(self.final_act(self.final_norm(h)))


class Hybrid2D1DFlowModel_SPATIOTEMPORAL_DUAL(nn.Module):
    """Release model: unet3d STMAP branch + dual-head velocity predictor."""

    def __init__(
        self,
        dim=64,
        dim_mults=(2,),
        stmap_channels=6,
        roi_n=4,
        c_channels=3,
        feature_dim=128,
        time_emb_dim=256,
        wavelet_input_channels=1,
        vp_num_down_layers=2,
        vp_base_dim=64,
        vp_dim_mults=(1, 2, 4),
        dual_head_fusion_hidden=32,
    ):
        super().__init__()

        if len(dim_mults) != 1:
            raise ValueError("Only a single dim_mult is supported in the release model.")

        fusion_dim = int(dim * dim_mults[0])
        if feature_dim != fusion_dim:
            raise ValueError(f"feature_dim ({feature_dim}) must equal dim * dim_mults[0] ({fusion_dim}).")

        self.dim = dim
        self.channels = 1
        self.init_dim = dim
        self.time_emb_dim = time_emb_dim
        self.stmap_channels = stmap_channels
        self.roi_n = roi_n
        self.roi_j = roi_n * roi_n
        self.c_channels = c_channels
        self.feature_dim = feature_dim

        self.initial_3d_conv_unet3d = Initial3DConv(stmap_channels, dim)
        self.spatiotemporal_fusion_unet3d = SpatioTemporalFusion3D(dim, feature_dim)
        self.spatial_pooling_unet3d = SpatialPooling(feature_dim, roi_n)

        self.condition_processor = ConditionProcessor(c_channels=c_channels, out_dim=feature_dim)

        self.stmap_velocity_predictor = ConvSimpleVelocityPredictor(
            input_channels=wavelet_input_channels,
            feature_dim=feature_dim,
            time_emb_dim=time_emb_dim,
            hidden_dim=feature_dim // 2,
        )
        self.c_velocity_predictor = AdaGNUNetVelocityPredictorV2(
            input_channels=wavelet_input_channels,
            feature_dim=feature_dim,
            time_emb_dim=time_emb_dim,
            base_dim=vp_base_dim,
            num_down_layers=vp_num_down_layers,
            dim_mults=vp_dim_mults,
            dropout=0.0,
        )
        self.dual_head_fusion = GatedFusionWithRefinement(
            time_emb_dim=time_emb_dim,
            hidden_dim=dual_head_fusion_hidden,
        )

        self.dual_head_stmap_norm = nn.GroupNorm(_group_count(feature_dim), feature_dim)
        self.dual_head_craw_norm = nn.GroupNorm(_group_count(feature_dim), feature_dim)
        self.null_condition = nn.Parameter(torch.zeros(1, feature_dim, 1))
        self.velocity_predictor = None

    def _get_time_embedding(self, time, device):
        half_dim = self.time_emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if emb.shape[-1] < self.time_emb_dim:
            emb = F.pad(emb, (0, self.time_emb_dim - emb.shape[-1]))
        return emb

    def _process_stmap(self, stmap):
        b, c, j, t = stmap.shape
        if c != self.stmap_channels:
            raise ValueError(f"Expected {self.stmap_channels} STMAP channels, got {c}.")
        if j != self.roi_j:
            raise ValueError(f"Expected {self.roi_j} ROI entries for roi_n={self.roi_n}, got {j}.")

        stmap_3d = stmap.reshape(b, c, self.roi_n, self.roi_n, t)
        h = self.initial_3d_conv_unet3d(stmap_3d)
        h = self.spatiotemporal_fusion_unet3d(h)
        h = self.spatial_pooling_unet3d(h)
        return h.squeeze(2).squeeze(2)

    def encode_conditions(self, stmap, c_raw):
        if c_raw.shape[1] != self.c_channels:
            raise ValueError(f"Expected {self.c_channels} traditional signal channels, got {c_raw.shape[1]}.")

        stmap_feat = self._process_stmap(stmap)
        craw_feat = self.condition_processor(c_raw)

        stmap_feat = self.dual_head_stmap_norm(stmap_feat)
        craw_feat = self.dual_head_craw_norm(craw_feat)
        return stmap_feat, craw_feat

    def forward(self, x_t, time, cond, context_mask, return_branch_outputs=False):
        if not isinstance(cond, (tuple, list)) or len(cond) != 2:
            raise ValueError("dual_head forward expects cond=(stmap_feat, craw_feat).")

        stmap_feat, craw_feat = cond
        batch_size = x_t.shape[0]
        time_emb = self._get_time_embedding(time, x_t.device)
        context_mask = context_mask.view(batch_size, 1, 1).to(dtype=x_t.dtype, device=x_t.device)
        null_cond = self.null_condition.expand(batch_size, -1, x_t.shape[-1])

        if stmap_feat.shape[-1] != x_t.shape[-1]:
            stmap_feat = F.interpolate(stmap_feat, size=x_t.shape[-1], mode="linear", align_corners=False)
        if craw_feat.shape[-1] != x_t.shape[-1]:
            craw_feat = F.interpolate(craw_feat, size=x_t.shape[-1], mode="linear", align_corners=False)

        stmap_feat = stmap_feat * (1 - context_mask) + null_cond * context_mask
        craw_feat = craw_feat * (1 - context_mask) + null_cond * context_mask

        v_stmap = self.stmap_velocity_predictor(stmap_feat, x_t, time_emb)
        v_c = self.c_velocity_predictor(craw_feat, x_t, time_emb)
        velocity, _, _ = self.dual_head_fusion(v_stmap, v_c, time_emb)

        if return_branch_outputs:
            return velocity, v_stmap, v_c
        return velocity


def _smoke_test():
    torch.manual_seed(0)

    batch_size = 2
    time_steps = 64
    roi_n = 4
    x_channels = 64

    model = Hybrid2D1DFlowModel_SPATIOTEMPORAL_DUAL(
        dim=64,
        dim_mults=(2,),
        stmap_channels=6,
        roi_n=roi_n,
        c_channels=3,
        feature_dim=128,
        time_emb_dim=256,
        wavelet_input_channels=x_channels,
        vp_num_down_layers=2,
        vp_base_dim=64,
        vp_dim_mults=(1, 2, 4),
        dual_head_fusion_hidden=32,
    )
    model.eval()

    stmap = torch.randn(batch_size, 6, roi_n * roi_n, time_steps)
    c_raw = torch.randn(batch_size, 3, time_steps)
    x_t = torch.randn(batch_size, x_channels, time_steps)
    time = torch.rand(batch_size)
    context_mask = torch.zeros(batch_size)

    with torch.no_grad():
        cond = model.encode_conditions(stmap, c_raw)
        velocity = model(x_t, time, cond, context_mask)
        branch_outputs = model(x_t, time, cond, context_mask, return_branch_outputs=True)

    print("cond:", cond[0].shape, cond[1].shape)
    print("velocity:", velocity.shape)
    print("branches:", [out.shape for out in branch_outputs])


if __name__ == "__main__":
    _smoke_test()
