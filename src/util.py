import math
import numpy as np
from constants import (
    HEIGHT,
    TENNIS_DATASET_ROOT,
    TRAIN_GAMES,
    VAL_GAMES,
    WIDTH,
)
from models.registry import build_model


def genHeatMap(w, h, cx, cy, r, mag, use_gaussian=False):
    """Render one heatmap target.

    A negative centre returns an all-zero map, which is how an invisible ball
    is encoded.  With ``use_gaussian`` the target is a Gaussian of radius
    ``r`` (``sigma`` in the dataset), otherwise a binary disk.  Only the
    bounding box is evaluated.
    """
    heatmap = np.zeros((h, w), dtype=np.float32)
    if cx < 0 or cy < 0:
        return heatmap

    box_r = int(math.ceil(r * 3)) if use_gaussian else int(math.ceil(r))

    x_min = max(0, int(cx) - box_r)
    x_max = min(w, int(cx) + box_r + 1)
    y_min = max(0, int(cy) - box_r)
    y_max = min(h, int(cy) + box_r + 1)

    if x_min >= x_max or y_min >= y_max:
        return heatmap

    x, y = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2

    if use_gaussian:
        g = np.exp(-dist_sq / (2 * (r ** 2)))
        heatmap[y_min:y_max, x_min:x_max] = g * mag
    else:
        heatmap[y_min:y_max, x_min:x_max][dist_sq <= r ** 2] = mag

    return heatmap


def get_model(model_name, height=HEIGHT, width=WIDTH, pretrained=True):
    """Instantiate a registered model by name.

    The roster lives in :mod:`models.registry`; see ``MODEL_NAMES`` for the
    accepted values.
    """
    return build_model(model_name, height, width, pretrained=pretrained)


def get_dataset(height=HEIGHT, width=WIDTH, augment=False, zoom_scales=(1.0,),
                train_games=None, val_games=None):
    """Return ``(train_dataset, val_dataset)`` for the game-level split.

    ``train_games`` / ``val_games`` override :data:`constants.TRAIN_GAMES` /
    :data:`constants.VAL_GAMES`; ``augment`` and ``zoom_scales`` apply to the
    training split only.
    """
    from dataset import TennisDataset

    train_games = list(train_games) if train_games else list(TRAIN_GAMES)
    val_games = list(val_games) if val_games else list(VAL_GAMES)

    def build(games, augment_split):
        return TennisDataset(
            root_dir=TENNIS_DATASET_ROOT,
            games=games,
            target_img_height=height,
            target_img_width=width,
            augment=augment_split,
            zoom_scales=zoom_scales if augment_split else (1.0,),
        )

    return build(train_games, bool(augment)), build(val_games, False)
