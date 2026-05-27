import torch
import torch.nn as nn
import torch.nn.functional as F


class STMAPConstructor(nn.Module):
    """Spatio-Temporal Map constructor: [B,3,T,H,W] -> [B,T,J,C]"""

    def __init__(self, roi_n=4, use_yuv=True):
        super(STMAPConstructor, self).__init__()
        self.roi_n = roi_n
        self.roi_j = roi_n * roi_n
        self.use_yuv = use_yuv
        self.n_channels = 6 if use_yuv else 3

    def rgb_to_yuv(self, rgb_tensor):
        rgb_to_yuv_matrix = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.14713, -0.28886, 0.436],
            [0.615, -0.51499, -0.10001]
        ], dtype=rgb_tensor.dtype, device=rgb_tensor.device)

        B, C, T, H, W = rgb_tensor.shape
        rgb_flat = rgb_tensor.permute(0, 2, 3, 4, 1).contiguous().view(-1, 3)
        yuv_flat = torch.matmul(rgb_flat, rgb_to_yuv_matrix.T)
        yuv_tensor = yuv_flat.view(B, T, H, W, 3).permute(0, 4, 1, 2, 3)

        return yuv_tensor

    def spatial_roi_pooling(self, video_tensor):
        B, C, T, H, W = video_tensor.shape

        video_reshaped = video_tensor.permute(0, 2, 1, 3, 4).contiguous()
        video_reshaped = video_reshaped.view(B * T, C, H, W)

        roi_frames = F.adaptive_avg_pool2d(video_reshaped, (self.roi_n, self.roi_n))
        roi_frames = roi_frames.view(B * T, C, self.roi_j)
        roi_features = roi_frames.view(B, T, C, self.roi_j).permute(0, 2, 1, 3)

        return roi_features

    def forward(self, video_input):
        B, input_C, T, H, W = video_input.shape
        assert input_C == 3
        assert H >= self.roi_n and W >= self.roi_n

        if self.use_yuv:
            yuv_tensor = self.rgb_to_yuv(video_input)
            rgbyuv_tensor = torch.cat([video_input, yuv_tensor], dim=1)
            roi_features = self.spatial_roi_pooling(rgbyuv_tensor)
        else:
            roi_features = self.spatial_roi_pooling(video_input)

        stmap = roi_features.permute(0, 2, 3, 1).contiguous()
        return stmap
