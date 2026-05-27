import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from torchdiffeq import odeint
from pytorch_wavelets import DWT1DForward, DWT1DInverse


class WaveletTransform:

    def __init__(self, wavelet='db4', level=4):
        self.wavelet = wavelet
        self.level = level
        self.dwt_forward = DWT1DForward(wave=wavelet, J=level, mode='reflect')
        self.dwt_inverse = DWT1DInverse(wave=wavelet, mode='reflect')

    def decompose(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.device != self.dwt_forward.h0.device:
            self.dwt_forward = self.dwt_forward.to(x.device)

        yl, yh = self.dwt_forward(x)

        coeffs = []
        coeffs.append(yl)
        for i in range(len(yh) - 1, -1, -1):
            coeffs.append(yh[i])

        return coeffs

    def reconstruct(self, coeffs: List[torch.Tensor]) -> torch.Tensor:
        yl = coeffs[0]

        if yl.device != self.dwt_inverse.g0.device:
            self.dwt_inverse = self.dwt_inverse.to(yl.device)

        yh = []
        for i in range(len(coeffs) - 1, 0, -1):
            yh.append(coeffs[i])

        x = self.dwt_inverse((yl, yh))
        return x



class WaveletFlowMatching1D(nn.Module):

    def __init__(self,
                 nn_model: nn.Module,
                 device: str,
                 wavelet: str = 'db4',
                 wavelet_level: int = 4,
                 scale_weights: Optional[List[float]] = None,
                 drop_prob: float = 0.1,
                 initial_wavelet_gamma: float = 1.0,
                 final_wavelet_gamma: float = 1.0,
                 gamma_transition_epochs: int = 50,
                 sigma_min: float = 1e-4,
                 wavelet_ratio: float = 1.0,
                 dual_head_loss_stmap: float = 0.3,
                 dual_head_loss_c: float = 0.3,
                 dual_head_loss_final: float = 0.4,
                 ):
        super().__init__()

        self.nn_model = nn_model.to(device)
        self.device = device
        self.drop_prob = drop_prob
        self.sigma_min = sigma_min
        self.wavelet_ratio = wavelet_ratio

        self.wavelet_transform = WaveletTransform(wavelet=wavelet, level=wavelet_level)
        self.wavelet_level = wavelet_level

        if scale_weights is None:
            if wavelet_level == 4:
                self.scale_weights = [2.0, 1.0, 1.0, 0.0, 0.0]
        else:
            self.scale_weights = scale_weights

        weight_sum = sum(self.scale_weights)
        self.scale_weights = [w / weight_sum for w in self.scale_weights]

        self.mse_loss = nn.MSELoss()

        self.dual_head_loss_stmap = dual_head_loss_stmap
        self.dual_head_loss_c = dual_head_loss_c
        self.dual_head_loss_final = dual_head_loss_final

        self.initial_wavelet_gamma = initial_wavelet_gamma
        self.final_wavelet_gamma = final_wavelet_gamma
        self.gamma_transition_epochs = max(1, gamma_transition_epochs)

        self.register_buffer('current_epoch', torch.tensor(0, dtype=torch.long))

    def update_current_epoch(self):
        self.current_epoch += 1

    def calculate_current_gammas(self):
        progress = min(self.current_epoch.item() / self.gamma_transition_epochs, 1.0)
        current_wavelet_gamma = (self.initial_wavelet_gamma +
                                 (self.final_wavelet_gamma - self.initial_wavelet_gamma) * progress)
        return current_wavelet_gamma

    def _compute_branch_loss(self, v_pred, u_target):
        v_pred_coeffs = self.wavelet_transform.decompose(v_pred)
        u_target_coeffs = self.wavelet_transform.decompose(u_target)

        loss_wavelet = 0.0
        for v_pred_coeff, u_target_coeff, weight in zip(v_pred_coeffs, u_target_coeffs, self.scale_weights):
            loss_wavelet += weight * self.mse_loss(v_pred_coeff, u_target_coeff)

        current_wavelet_gamma = self.calculate_current_gammas()
        loss_total = current_wavelet_gamma * self.wavelet_ratio * loss_wavelet

        return loss_total

    def forward(self, x1_real: torch.Tensor, c_raw: torch.Tensor) -> torch.Tensor:
        batch_size = x1_real.shape[0]

        t_scalar = torch.rand(batch_size, device=self.device) * (1.0 - self.sigma_min * 2) + self.sigma_min
        t_reshaped = t_scalar.view(-1, 1, 1)
        x0_prior = torch.randn_like(x1_real)

        x_t = (1 - t_reshaped) * x0_prior + t_reshaped * x1_real
        u_target_time = x1_real - x0_prior

        context_mask = (torch.rand(batch_size, device=self.device) < self.drop_prob).float()

        v_pred_final, v_pred_stmap, v_pred_c = self.nn_model(
            x_t, t_scalar, c_raw, context_mask, return_branch_outputs=True
        )

        loss_final = self._compute_branch_loss(v_pred_final, u_target_time)
        loss_stmap = self._compute_branch_loss(v_pred_stmap, u_target_time)
        loss_c = self._compute_branch_loss(v_pred_c, u_target_time)

        loss_total = (self.dual_head_loss_stmap * loss_stmap +
                      self.dual_head_loss_c * loss_c +
                      self.dual_head_loss_final * loss_final)

        return loss_total

    def sample_ode_torchdiffeq(self,
                               n_sample: int,
                               length: int,
                               c_i: torch.Tensor,
                               guide_w: float = 5.0,
                               num_eval_points: int = 50,
                               ode_method: str = 'dopri5',
                               rtol: float = 1e-5,
                               atol: float = 1e-5
                               ) -> Tuple[torch.Tensor, Optional[np.ndarray], int]:
        self.nn_model.eval()

        x_0 = torch.randn(n_sample, self.nn_model.channels, length, device=self.device)

        t_span = torch.linspace(0.0, 1.0, num_eval_points, device=self.device)

        is_dual_head = isinstance(c_i, tuple)
        if is_dual_head:
            c_i_eff = (c_i[0].to(self.device), c_i[1].to(self.device))
            null_c = (torch.zeros_like(c_i_eff[0]), torch.zeros_like(c_i_eff[1]))
        else:
            c_i_eff = c_i.to(self.device)
            null_c = torch.zeros_like(c_i_eff, device=self.device)

        zeros_mask = torch.zeros(n_sample, device=self.device)
        ones_mask = torch.ones(n_sample, device=self.device)

        nfe_counter = 0

        def ode_dynamics_func(t_scalar, x_t_current):
            nonlocal nfe_counter
            nfe_counter += 1

            t_batch_single = torch.full((x_t_current.shape[0],),
                                        t_scalar.item(),
                                        device=self.device)

            with torch.no_grad():
                x_t_batch = torch.cat([x_t_current, x_t_current], dim=0)
                t_batch = torch.cat([t_batch_single, t_batch_single], dim=0)
                if is_dual_head:
                    c_batch = (
                        torch.cat([c_i_eff[0], null_c[0]], dim=0),
                        torch.cat([c_i_eff[1], null_c[1]], dim=0)
                    )
                else:
                    c_batch = torch.cat([c_i_eff, null_c], dim=0)
                mask_batch = torch.cat([zeros_mask, ones_mask], dim=0)

                v_both = self.nn_model(x_t_batch, t_batch, c_batch, mask_batch)

                v_cond = v_both[:n_sample]
                v_uncond = v_both[n_sample:]

                v_pred = (1 + guide_w) * v_cond - guide_w * v_uncond

            return v_pred

        try:
            solution_trajectory = odeint(
                ode_dynamics_func,
                x_0,
                t_span,
                method=ode_method,
                rtol=rtol,
                atol=atol
            )
        except Exception as e:
            solution_trajectory = odeint(
                ode_dynamics_func,
                x_0,
                t_span,
                method='euler',
                rtol=1e-3,
                atol=1e-3
            )

        x_t_final = solution_trajectory[-1]
        x_t_store_np = solution_trajectory.cpu().numpy() if num_eval_points > 1 else None

        return x_t_final, x_t_store_np, nfe_counter


class CrossScaleAttention(nn.Module):

    def __init__(self, feature_dim: int, num_scales: int = 6, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_scales = num_scales
        self.num_heads = num_heads

        self.scale_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout)
        )

    def forward(self, scale_features: list) -> list:
        B = scale_features[0].shape[0]
        C = scale_features[0].shape[1]
        L = scale_features[0].shape[2]

        stacked = torch.stack(scale_features, dim=1)
        stacked = stacked.permute(0, 3, 1, 2)
        stacked = stacked.reshape(B * L, self.num_scales, C)

        residual = stacked
        stacked = self.norm1(stacked)
        attn_out, _ = self.scale_attn(stacked, stacked, stacked)
        stacked = residual + attn_out

        residual = stacked
        stacked = self.norm2(stacked)
        stacked = residual + self.ffn(stacked)

        stacked = stacked.reshape(B, L, self.num_scales, C)
        stacked = stacked.permute(0, 2, 3, 1)

        attended_features = [stacked[:, i, :, :] for i in range(self.num_scales)]

        return attended_features


class WaveletEnhancedUNet1D(nn.Module):

    @staticmethod
    def calculate_output_channels(level: int, init_dim: int = 64, channels: int = 1) -> int:
        return 64

    def __init__(self, base_unet: nn.Module, wavelet: str = 'db4', level: int = 4, init_dim: int = 64):
        super().__init__()

        self.base_unet = base_unet
        self.wavelet_transform = WaveletTransform(wavelet=wavelet, level=level)
        self.level = level

        self.channels = 1
        self.init_dim = init_dim

        self.scale_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.channels, self.init_dim // 4, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(self.init_dim // 4, self.init_dim // 4, 3, padding=1),
                nn.ReLU()
            ) for _ in range(level + 1)
        ])

        self.wavelet_norms = nn.ModuleList([
            nn.InstanceNorm1d(
                num_features=1,
                eps=1e-8,
                affine=True,
                track_running_stats=False
            )
            for _ in range(level + 1)
        ])

        self.output_channels = self.calculate_output_channels(level, self.init_dim, self.channels)

        feature_dim = self.init_dim // 4
        num_scales = level + 1

        self.xt_encoder = nn.Sequential(
            nn.Conv1d(self.channels, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.cross_scale_attention = CrossScaleAttention(
            feature_dim=feature_dim,
            num_scales=num_scales + 1,
            num_heads=4,
            dropout=0.1
        )

        total_input_channels = feature_dim * (num_scales + 1) + self.channels

        self.output_projection = nn.Sequential(
            nn.Conv1d(total_input_channels, self.output_channels, kernel_size=1),
            nn.GroupNorm(min(8, self.output_channels), self.output_channels),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor, c: torch.Tensor,
                context_mask: torch.Tensor, return_branch_outputs: bool = False) -> torch.Tensor:

        wavelet_coeffs = self.wavelet_transform.decompose(x)

        normalized_coeffs = [
            norm(coeffs)
            for coeffs, norm in zip(wavelet_coeffs, self.wavelet_norms)
        ]

        scale_features = []
        for i, (coeffs, extractor) in enumerate(zip(normalized_coeffs, self.scale_extractors)):
            if coeffs.shape[2] != x.shape[2]:
                coeffs_resized = F.interpolate(
                    coeffs,
                    size=x.shape[2],
                    mode='linear',
                    align_corners=False
                )
            else:
                coeffs_resized = coeffs

            features = extractor(coeffs_resized)
            scale_features.append(features)

        xt_feat = self.xt_encoder(x)
        all_features = [xt_feat] + scale_features
        attended_features = self.cross_scale_attention(all_features)

        fused_wavelet = torch.cat(attended_features, dim=1)
        combined = torch.cat([x, fused_wavelet], dim=1)
        x_enhanced = self.output_projection(combined)

        output = self.base_unet(x_enhanced, time, c, context_mask, return_branch_outputs=return_branch_outputs)

        return output
