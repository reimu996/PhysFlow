import torch
from scipy import signal


def CHROM_torch_batch(C, fps):
    if not isinstance(C, torch.Tensor):
        C = torch.tensor(C, dtype=torch.float32)

    l = int(round(1.6 * fps))
    batch_size, _, N = C.shape
    H = torch.zeros(batch_size, N, dtype=C.dtype, device=C.device)

    for n in range(N + 1):
        m = n - l
        if m > 0:
            Cn = C[:, :, m:n] / torch.mean(C[:, :, m:n], dim=2, keepdim=True) - 1

            S = torch.tensor([[3, -2, 0], [1.5, 1, -1.5]], dtype=C.dtype, device=C.device).unsqueeze(0)
            S = S.repeat(batch_size, 1, 1)
            S = torch.bmm(S, Cn)

            std_0 = torch.std(S[:, 0, :], unbiased=False, dim=1)
            std_1 = torch.std(S[:, 1, :], unbiased=False, dim=1)

            h = S[:, 0, :] - (std_0 / std_1).unsqueeze(1) * S[:, 1, :]

            win_len = h.shape[-1]
            window = torch.hann_window(win_len, periodic=False, dtype=C.dtype, device=C.device)
            window = window.unsqueeze(0).repeat(batch_size, 1)
            h = h * window

            H[:, m:n] = H[:, m:n] + h

    return H


def POS_torch_batch(C, fps):
    if not isinstance(C, torch.Tensor):
        C = torch.tensor(C, dtype=torch.float32)

    l = int(round(1.6 * fps))
    batch_size, _, N = C.shape
    H = torch.zeros(batch_size, N, dtype=C.dtype, device=C.device)

    for n in range(N + 1):
        m = n - l
        if m >= 0:
            Cn = C[:, :, m:n] / torch.mean(C[:, :, m:n], dim=2, keepdim=True) - 1
            S = torch.tensor([[0, 1, -1], [-2, 1, 1]], dtype=C.dtype, device=C.device).unsqueeze(0)
            S = S.repeat(batch_size, 1, 1)
            S = torch.bmm(S, Cn)
            std_0 = torch.std(S[:, 0, :], dim=1, unbiased=False)
            std_1 = torch.std(S[:, 1, :], dim=1, unbiased=False)
            h = S[:, 0, :] + (std_0 / (std_1 + 1e-10)).unsqueeze(1) * S[:, 1, :]
            h = (h - torch.mean(h, dim=1, keepdim=True)) / (torch.std(h, dim=1, keepdim=True, unbiased=False) + 1e-10)
            H[:, m:n] = H[:, m:n] + h

    return H


def PBV_torch_batch(C, fps):
    if not isinstance(C, torch.Tensor):
        C = torch.tensor(C, dtype=torch.float32)

    l = int(round(1.6 * fps))
    batch_size, _, N = C.shape
    H = torch.zeros(batch_size, N, dtype=C.dtype, device=C.device)

    u_pbv = torch.tensor([0.33, 0.77, 0.53], dtype=C.dtype, device=C.device).reshape(1, 1, 3)

    for n in range(N + 1):
        m = n - l
        if m > 0:
            Cn = C[:, :, m:n] / torch.mean(C[:, :, m:n], dim=2, keepdim=True) - 1
            CnT = Cn.transpose(1, 2)
            Cov = Cn @ CnT
            Cov_inv = torch.inverse(Cov + 1e-5 * torch.eye(3, device=C.device).unsqueeze(0))
            h = u_pbv @ Cov_inv @ Cn
            h = h.squeeze(1)

            hann_window = torch.tensor(signal.windows.hann(h.shape[1]), dtype=C.dtype, device=C.device)
            h = h * hann_window.unsqueeze(0)

            H[:, m:n] = H[:, m:n] + h

    return H
