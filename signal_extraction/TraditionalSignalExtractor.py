import torch
import torch.nn as nn
from signal_extraction.traditional_rppg_methods import CHROM_torch_batch, POS_torch_batch, PBV_torch_batch
from signal_extraction.roi_extraction import extract_roi_mean


class TraditionalSignalExtractor(nn.Module):
    """Extract traditional rPPG signals (CHROM, POS, PBV) from video.

    Input: video [B, C, T, H, W]
    Output: c_raw [B, num_signals, T]
    """

    def __init__(self, signals=['CHROM', 'POS', 'PBV'], roi_type='full'):
        super().__init__()

        self.signals = signals
        self.roi_type = roi_type
        self.num_signals = len(signals)

        self.method_map = {
            'CHROM': CHROM_torch_batch,
            'POS': POS_torch_batch,
            'PBV': PBV_torch_batch
        }

        for sig in signals:
            if sig not in self.method_map:
                raise ValueError(f"Unsupported signal type: {sig}")

    def forward(self, video, fps=30, batch_count=0):
        B, C, T, H, W = video.shape

        mean_imgs = extract_roi_mean(video, batch_count)

        rPPG_list = []

        for method_name in self.signals:
            rPPG_raw = self.method_map[method_name](mean_imgs, fps)

            rPPG_mean = torch.mean(rPPG_raw, dim=1, keepdim=True)
            rPPG_std = torch.std(rPPG_raw, dim=1, keepdim=True)
            rPPG_normalized = (rPPG_raw - rPPG_mean) / (rPPG_std + 1e-8)

            rPPG_list.append(rPPG_normalized.unsqueeze(1))

        c_raw = torch.cat(rPPG_list, dim=1)

        return c_raw
