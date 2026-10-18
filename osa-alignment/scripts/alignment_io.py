"""Image measurements, replay manifests, and JSON output for OSA alignment."""
import json
from pathlib import Path

import numpy as np

from evaluate_alignment import Config, evaluate, json_safe, read_image


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def load_manifest(path):
    """Load positions and image paths; positions use the manifest's explicit units."""
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if not isinstance(data.get("units"), str) or not data["units"].strip():
        raise ValueError("Manifest must specify motor units")
    if not data.get("samples"):
        raise ValueError("Manifest samples must not be empty")
    for row in data["samples"]:
        p = np.asarray(row["position"], dtype=float)
        if p.shape != (2,) or not np.isfinite(p).all():
            raise ValueError("Each position must contain two finite numbers")
        row["position"] = p.tolist()
        row["image"] = str((path.parent / row["image"]).resolve())
    return data


def measure(path, *, config=None):
    image = read_image(path)
    result = evaluate(image, Config() if config is None else config)
    appearance = {}
    if result.get("fit") is not None:
        fit = result["fit"]
        y, x = np.indices(image.shape)
        dx, dy = x - fit["center_x"], y - fit["center_y"]
        angle = np.deg2rad(fit["angle_degrees"])
        u = dx * np.cos(angle) + dy * np.sin(angle)
        v = -dx * np.sin(angle) + dy * np.cos(angle)
        inner, outer = fit["inner_axes"], fit["outer_axes"]
        hole = (u / inner[0]) ** 2 + (v / inner[1]) ** 2 <= 1
        annulus = ((u / outer[0]) ** 2 + (v / outer[1]) ** 2 <= 1) & ~hole
        sectors = {"left": (dx < 0) & (abs(dx) >= abs(dy)),
                   "right": (dx >= 0) & (abs(dx) >= abs(dy)),
                   "top": (dy < 0) & (abs(dy) > abs(dx)),
                   "bottom": (dy >= 0) & (abs(dy) > abs(dx))}
        appearance = {"ring_sector_means": {
            name: float(image[mask & annulus].mean()) if (mask & annulus).any() else None
            for name, mask in sectors.items()},
            "hole_peak": float(image[hole].max()) if hole.any() else None}
    return {"image": str(path), "metrics": result,
            "appearance": appearance}
