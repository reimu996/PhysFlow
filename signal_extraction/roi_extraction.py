import torch
import numpy as np
import time
import os
import cv2

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False


def extract_roi_mean(X_batch, batch_count, save_dir=None, max_detect_frames=None):
    """Extract ROI mean from video batch using MTCNN face detection.

    Args:
        X_batch: [N, C, T, H, W]
    Returns:
        mean_imgs: [N, C, T]
    """
    if not MTCNN_AVAILABLE:
        mean_imgs = X_batch.mean(dim=(3, 4))
        return mean_imgs

    N, C, T, H, W = X_batch.shape
    device = X_batch.device

    mtcnn = MTCNN(device=device)
    mean_imgs_list = []

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    if max_detect_frames is None:
        max_detect_frames = T
    else:
        max_detect_frames = min(max_detect_frames, T)

    first_frames = X_batch[:, :, 0, :, :].permute(0, 2, 3, 1)
    first_frames_np = first_frames.cpu().numpy()

    if np.max(first_frames_np) <= 1.0 and np.min(first_frames_np) >= 0.0:
        first_frames_np = (first_frames_np * 255).astype(np.uint8)
    else:
        first_frames_np = first_frames_np.astype(np.uint8)

    boxes, probs, landmarks = mtcnn.detect(first_frames_np, landmarks=True)

    detection_results = {}
    failed_samples = []

    for n in range(N):
        if (boxes[n] is not None and len(boxes[n]) > 0 and
                landmarks[n] is not None and len(landmarks[n]) > 0):
            try:
                landmarks_array = np.array(landmarks[n][0], dtype=np.float32)
                if np.isnan(landmarks_array).any():
                    failed_samples.append(n)
                else:
                    detection_results[n] = {
                        'landmarks': landmarks[n][0],
                        'frame_idx': 0
                    }
            except (ValueError, TypeError):
                failed_samples.append(n)
        else:
            failed_samples.append(n)

    if len(failed_samples) > 0 and max_detect_frames > 1:
        for frame_idx in range(1, max_detect_frames):
            if len(failed_samples) == 0:
                break

            still_failed_samples = []

            if len(failed_samples) > 0:
                failed_indices = torch.tensor(failed_samples, device=device)
                current_frames_batch = X_batch[failed_indices, :, frame_idx, :, :].permute(0, 2, 3, 1)
                current_frames_np = current_frames_batch.cpu().numpy()

                if np.max(current_frames_np) <= 1.0 and np.min(current_frames_np) >= 0.0:
                    current_frames_np = (current_frames_np * 255).astype(np.uint8)
                else:
                    current_frames_np = current_frames_np.astype(np.uint8)

                boxes_batch, probs_batch, landmarks_batch = mtcnn.detect(current_frames_np, landmarks=True)

                for i, n in enumerate(failed_samples):
                    if (boxes_batch[i] is not None and len(boxes_batch[i]) > 0 and
                            landmarks_batch[i] is not None and len(landmarks_batch[i]) > 0):
                        try:
                            landmarks_array = np.array(landmarks_batch[i][0], dtype=np.float32)
                            if np.isnan(landmarks_array).any():
                                still_failed_samples.append(n)
                            else:
                                detection_results[n] = {
                                    'landmarks': landmarks_batch[i][0],
                                    'frame_idx': frame_idx
                                }
                        except (ValueError, TypeError):
                            still_failed_samples.append(n)
                    else:
                        still_failed_samples.append(n)

            failed_samples = still_failed_samples

    for n in range(N):
        use_default_roi = False

        if n in detection_results:
            landmarks_n = detection_results[n]['landmarks']

            left_eye = landmarks_n[0]
            right_eye = landmarks_n[1]
            nose_tip = landmarks_n[2]

            eyes_distance = np.linalg.norm(left_eye - right_eye)
            forehead_height = 0.5 * eyes_distance
            horizontal_extension = 0.2 * eyes_distance

            x_min = int(left_eye[0] - horizontal_extension)
            x_max = int(right_eye[0] + horizontal_extension)
            y_max_eye = int(min(left_eye[1], right_eye[1]))
            y_min = int(y_max_eye - forehead_height)
            y_max = int(nose_tip[1])

            x_min = max(x_min, 0)
            x_max = min(x_max, W)
            y_min = max(y_min, 0)
            y_max = min(y_max, H)

            if x_min >= x_max or y_min >= y_max:
                use_default_roi = True
            else:
                roi_area = (y_max - y_min) * (x_max - x_min)
                if roi_area < 300:
                    use_default_roi = True
        else:
            use_default_roi = True

        if use_default_roi:
            crop_h = max(1, int(H * 0.6))
            crop_w = max(1, int(W * 0.6))
            start_h = (H - crop_h) // 2
            start_w = (W - crop_w) // 2
            x_min, x_max = start_w, start_w + crop_w
            y_min, y_max = start_h, start_h + crop_h

        roi_data = X_batch[n, :, :, y_min:y_max, x_min:x_max]
        mean_imgs_n = roi_data.mean(dim=(-2, -1))
        mean_imgs_list.append(mean_imgs_n)

        if save_dir is not None:
            img = first_frames_np[n]
            if not img.flags['C_CONTIGUOUS']:
                img = np.ascontiguousarray(img)
            img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            save_path = os.path.join(save_dir, f"batch_{batch_count}_sample_{n}_roi.png")
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    mean_imgs = torch.stack(mean_imgs_list, dim=0)

    return mean_imgs
