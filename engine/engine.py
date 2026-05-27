from engine.engine_tools import *
from tqdm import tqdm
import os
import torch
import numpy as np


def train_epoch(flow_matcher, train_loader, fps,
                optimizer_G, scheduler_G,
                device, epoch,
                stmap_constructor,
                signal_extractor):
    accum_loss = 0.0
    batch_count = 0
    flow_matcher.train()

    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training", leave=False):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        X_batch = X_batch.float() / 255.0

        stmap = stmap_constructor(X_batch)
        stmap = stmap.permute(0, 3, 2, 1)

        c_raw = signal_extractor(X_batch, fps=fps, batch_count=batch_count)

        fused_cond = flow_matcher.nn_model.base_unet.encode_conditions(stmap, c_raw)

        y_batch_mean = torch.mean(y_batch, dim=1, keepdim=True)
        y_batch_std = torch.std(y_batch, dim=1, keepdim=True)
        y_batch_normalized = (y_batch - y_batch_mean) / (y_batch_std + 1e-8)
        real_bvp = y_batch_normalized.unsqueeze(1)

        optimizer_G.zero_grad()

        flow_matching_loss = flow_matcher(real_bvp, fused_cond)

        flow_matching_loss.backward()

        torch.nn.utils.clip_grad_norm_(flow_matcher.parameters(), max_norm=0.5)
        optimizer_G.step()

        current_loss = flow_matching_loss.item()
        accum_loss += current_loss
        batch_count += 1

    flow_matcher.update_current_epoch()

    avg_epoch_loss = accum_loss / batch_count if batch_count > 0 else 0.0
    print(f"Epoch Avg Loss: {avg_epoch_loss:.4f}")
    scheduler_G.step()

    return avg_epoch_loss



def test_epoch(flow_matcher, test_loader, fps, device,
               stmap_constructor, signal_extractor,
               guide_w=5.0, num_eval_points=50, ode_method='dopri5',
               rtol=1e-5, atol=1e-5):
    flow_matcher.eval()
    batch_count = 0

    all_hr_preds = []
    all_hr_labels = []

    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Testing", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            X_batch = X_batch.float() / 255.0

            stmap = stmap_constructor(X_batch)
            stmap = stmap.permute(0, 3, 2, 1)
            c_raw = signal_extractor(X_batch, fps=fps, batch_count=batch_count)

            fused_cond = flow_matcher.nn_model.base_unet.encode_conditions(stmap, c_raw)

            y_batch_mean = torch.mean(y_batch, dim=1, keepdim=True)
            y_batch_std = torch.std(y_batch, dim=1, keepdim=True)
            y_batch_normalized = (y_batch - y_batch_mean) / (y_batch_std + 1e-8)
            real_bvp = y_batch_normalized.unsqueeze(1)

            length = c_raw.shape[-1]

            generated_samples, generated_samples_history, nfe = flow_matcher.sample_ode_torchdiffeq(
                n_sample=X_batch.shape[0],
                length=length,
                c_i=fused_cond,
                guide_w=guide_w,
                num_eval_points=num_eval_points,
                ode_method=ode_method,
                rtol=rtol,
                atol=atol,
            )

            pred_bvp_normalized = generated_samples.squeeze(1)

            refined_rppg = pred_bvp_normalized
            refined_rppg_mean = torch.mean(refined_rppg, dim=1, keepdim=True)
            refined_rppg_std = torch.std(refined_rppg, dim=1, keepdim=True)
            refined_rppg_normalized = (refined_rppg - refined_rppg_mean) / (refined_rppg_std + 1e-8)

            gt_rppg_normalized = real_bvp.squeeze(1)

            batch_count += 1

            hr_preds, hr_labels = get_batch_hr_preds_and_labels(refined_rppg_normalized, gt_rppg_normalized, fps=fps)
            all_hr_preds.append(hr_preds)
            all_hr_labels.append(hr_labels)

    all_hr_preds = np.concatenate(all_hr_preds, axis=0)
    all_hr_labels = np.concatenate(all_hr_labels, axis=0)
    metrics = evaluate_predictions(all_hr_preds, all_hr_labels)

    pearson_corr = metrics["Pearson Correlation"]
    rmse = metrics["RMSE"]
    mae = metrics["MAE"]

    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, Pearson: {pearson_corr:.4f}")

    return rmse, pearson_corr, mae



def save_train_records_to_txt(records, save_dir, filename=None):
    os.makedirs(save_dir, exist_ok=True)

    if not filename:
        filename = "train_info.txt"
    else:
        filename = f"{filename}.txt"

    save_path = os.path.join(save_dir, filename)

    if len(records) == 0:
        return

    content = "| Epoch | Train Loss |\n|-------|------------|\n"

    for epoch in sorted(records.keys()):
        metrics = records[epoch]
        content += f"| {epoch + 1:>5d} | {metrics['train_loss']:>10.4f} |\n"

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(content)
