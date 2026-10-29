"""Crop the dashboard screenshots into the figures used by Chapter 5."""
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
S, F = HERE / "shots", HERE / "figures"

crops = {"ui_live.png": ("live.png", (630, 360, 2210, 1380)),
         "ui_gate.png": ("gate.png", (30, 360, 1490, 1230)),
         "ui_hdfs.png": ("hdfs.png", (30, 360, 1490, 1460))}
for out, (src, box) in crops.items():
    Image.open(S / src).convert("RGB").crop(box).save(F / out)
    print(out, Image.open(F / out).size)
