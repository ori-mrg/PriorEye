import os
import pickle
from typing import Any, Dict, List

import numpy as np


def _normalize_angle_rad(angle: float) -> float:
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def draw_arrow(ax, x, y, yaw, length=1.0, color="r"):
    dx = length * np.cos(yaw)
    dy = length * np.sin(yaw)
    ax.arrow(
        x, y, dx, dy,
        head_width=0.2 * length,
        head_length=0.3 * length,
        fc=color,
        ec=color,
    )


def load_data(file_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    print(f"Successfully loaded {len(data)} lane/connector records.")
    return data
