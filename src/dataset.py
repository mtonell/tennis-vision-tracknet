"""On-the-fly tennis dataset: 3-frame windows and 3 heatmap targets.

A sample is three consecutive frames stacked into ``(9, H, W)`` plus the
matching ``(3, H, W)`` Gaussian heatmaps.  Frames are resized on the fly.
Training also applies a per-window flip and zoom crop; a ball cropped out of
frame becomes an invisible target.
"""

import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from constants import HEIGHT, MAG, SIGMA, WIDTH
from util import genHeatMap


class TennisDataset(Dataset):
    """A tennis dataset indexed by whole games.

    ``augment`` enables flip and zoom and belongs to the training split only.
    ``zoom_scales`` is the set of crop scales sampled per window; ``(1.0,)``
    disables zoom.
    """

    def __init__(self, root_dir, games, target_img_height=HEIGHT,
                 target_img_width=WIDTH, sequence_dim=(3, 3), mag=MAG,
                 sigma=SIGMA, augment=False, flip_p=0.5,
                 zoom_scales=(1.0,)):
        self.root_dir = root_dir
        self.games = list(games)
        self.target_img_height = target_img_height
        self.target_img_width = target_img_width
        self.sequence_dim = sequence_dim
        self.mag = mag
        self.sigma = sigma
        self.augment = augment
        self.flip_p = flip_p
        self.zoom_scales = tuple(zoom_scales) if zoom_scales else (1.0,)

        self.samples = []
        self._build_index()

    def _build_index(self):
        for game in self.games:
            game_folder = os.path.join(self.root_dir, "Dataset", game)
            if not os.path.isdir(game_folder):
                continue
            for clip in sorted(os.listdir(game_folder)):
                clip_folder = os.path.join(game_folder, clip)
                if os.path.isdir(clip_folder):
                    self._add_clip(clip_folder)

    def _add_clip(self, clip_folder):
        label_path = os.path.join(clip_folder, 'Label.csv')
        if not os.path.exists(label_path):
            return

        label_data = pd.read_csv(label_path)
        file_names = label_data['file name'].values
        visibility = label_data['visibility'].values
        x_coords = label_data['x-coordinate'].values
        y_coords = label_data['y-coordinate'].values

        if len(file_names) == 0:
            return

        sample_img_path = os.path.join(clip_folder, str(file_names[0]))
        if not os.path.exists(sample_img_path):
            return

        # Label scale is original-frame height over model height, per clip.
        with Image.open(sample_img_path) as sample_img:
            ratio = sample_img.height / self.target_img_height

        num_frames = len(file_names)
        for i in range(num_frames - (self.sequence_dim[0] - 1)):
            frames_exist = all(
                os.path.exists(os.path.join(clip_folder, str(file_names[i + j])))
                for j in range(self.sequence_dim[0])
            )
            if not frames_exist:
                continue

            self.samples.append({
                'clip_folder': clip_folder,
                'file_names': file_names[i:i + self.sequence_dim[0]].tolist(),
                'visibility': visibility[i:i + self.sequence_dim[0]].tolist(),
                'x_coords': x_coords[i:i + self.sequence_dim[0]].tolist(),
                'y_coords': y_coords[i:i + self.sequence_dim[0]].tolist(),
                'ratio': ratio,
            })

    def __len__(self):
        return len(self.samples)

    def _sample_crop(self):
        """Sample one zoom-in crop for this sample.

        Returns ``(x0, y0, crop_w, crop_h, scale_x, scale_y)`` or ``None``.
        One crop for all three frames keeps the motion cue intact.
        """
        if not self.augment or not self.zoom_scales:
            return None
        scale = float(np.random.choice(self.zoom_scales))
        if scale >= 1.0:
            return None
        crop_w = max(1, int(round(self.target_img_width * scale)))
        crop_h = max(1, int(round(self.target_img_height * scale)))
        x0 = int(np.random.randint(0, self.target_img_width - crop_w + 1))
        y0 = int(np.random.randint(0, self.target_img_height - crop_h + 1))
        return (x0, y0, crop_w, crop_h,
                crop_w / self.target_img_width, crop_h / self.target_img_height)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        flip = self.augment and np.random.random() < self.flip_p
        crop = self._sample_crop()

        frames_sequence = []
        for file_name in sample['file_names']:
            frame_path = os.path.join(sample['clip_folder'], str(file_name))

            img = cv2.imread(frame_path)
            if img is None:
                img = np.zeros(
                    (self.target_img_height, self.target_img_width, 3),
                    dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if (img.shape[0] != self.target_img_height
                        or img.shape[1] != self.target_img_width):
                    img = cv2.resize(
                        img, (self.target_img_width, self.target_img_height))
                if crop is not None:
                    x0, y0, crop_w, crop_h, _, _ = crop
                    img = img[y0:y0 + crop_h, x0:x0 + crop_w]
                    img = cv2.resize(
                        img, (self.target_img_width, self.target_img_height))
                if flip:
                    img = np.ascontiguousarray(img[:, ::-1])

            img_array = img.astype(np.float32) / 255.0
            img_array = np.moveaxis(img_array, -1, 0)
            frames_sequence.extend([img_array[0], img_array[1], img_array[2]])

        x_data = np.stack(frames_sequence, axis=0)  # (9, H, W)

        heatmap_sequence = []
        for j in range(self.sequence_dim[1]):
            visible = sample['visibility'][j] != 0
            cx = sample['x_coords'][j] / sample['ratio']
            cy = sample['y_coords'][j] / sample['ratio']

            if visible and crop is not None:
                x0, y0, _, _, scale_x, scale_y = crop
                cx = (cx - x0) / scale_x
                cy = (cy - y0) / scale_y
                # Ball left the crop, so there is no valid label.
                if not (0 <= cx <= self.target_img_width - 1
                        and 0 <= cy <= self.target_img_height - 1):
                    visible = False

            if visible and flip:
                cx = self.target_img_width - 1 - cx

            if not visible:
                heatmap = genHeatMap(
                    self.target_img_width, self.target_img_height,
                    -1, -1, self.sigma, self.mag, use_gaussian=True)
            else:
                heatmap = genHeatMap(
                    self.target_img_width, self.target_img_height,
                    int(cx), int(cy), self.sigma, self.mag, use_gaussian=True)
            heatmap_sequence.append(heatmap)

        y_data = np.stack(heatmap_sequence, axis=0)  # (3, H, W)

        return torch.from_numpy(x_data).float(), torch.from_numpy(y_data).float()
