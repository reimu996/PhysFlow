import numpy as np
import torch
from scipy.stats import pearsonr

from engine.utils_sig import *


def get_batch_hr_preds_and_labels(rppg, y_batch, fps=30):
    if isinstance(rppg, torch.Tensor):
        rppg = rppg.detach().cpu().numpy()
    if isinstance(y_batch, torch.Tensor):
        y_batch = y_batch.detach().cpu().numpy()

    batch_size = rppg.shape[0]
    fs = fps
    lowcut = 0.6
    highcut = 4.0

    hr_preds = []
    hr_labels = []

    for i in range(batch_size):
        ri = rppg[i]
        yi = y_batch[i]

        filtered_ri = butter_bandpass(ri, lowcut, highcut, fs, order=2)
        filtered_yi = butter_bandpass(yi, lowcut, highcut, fs, order=2)

        hr_pred = hr_fft(filtered_ri, fs, harmonics_removal=True)
        hr_label = hr_fft(filtered_yi, fs, harmonics_removal=True)

        hr_preds.append(hr_pred)
        hr_labels.append(hr_label)

    return hr_preds, hr_labels


def evaluate_predictions(all_hr_preds, all_hr_labels):
    pearson_corr, _ = pearsonr(all_hr_preds, all_hr_labels)
    rmse = np.sqrt(np.mean((all_hr_preds - all_hr_labels) ** 2))
    mae = np.mean(np.abs(all_hr_preds - all_hr_labels))

    return {
        "Pearson Correlation": pearson_corr,
        "RMSE": rmse,
        "MAE": mae
    }
