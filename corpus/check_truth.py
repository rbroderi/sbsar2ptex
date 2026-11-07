"""Summarize truth renders: size, mode, per-channel min/max/mean, and whether
the output equals the main input pattern (a node that did nothing).

    python check_truth.py ROOT [SUBSTR]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(sys.argv[1])
match = sys.argv[2] if len(sys.argv) > 2 else ""


def load(p: Path) -> np.ndarray:
    a = np.asarray(Image.open(p))
    return a.astype(np.float64) / (65535.0 if a.dtype == np.uint16 else 255.0)


pats = {n: load(root / "patterns" / f"{n}.png") for n in ("coded_rgba", "coded_gray16", "noise_rgba", "noise_gray")}
for d in sorted((root / "truth").iterdir()):
    if match not in d.name:
        continue
    for f in sorted(d.glob("*.png")):
        img = Image.open(f)
        a = load(f)
        chans = a.reshape(a.shape[0], a.shape[1], -1)
        stats = " ".join(f"{c.min():.3f}/{c.max():.3f}/{c.mean():.3f}" for c in np.moveaxis(chans, 2, 0))
        same = ""
        for n, pa in pats.items():
            pc = pa.reshape(pa.shape[0], pa.shape[1], -1)
            if pc.shape[:2] == chans.shape[:2]:
                k = min(pc.shape[2], chans.shape[2], 3)
                err = np.abs(pc[..., :k] - chans[..., :k]).max()
                if err < 2 / 255:
                    same += f" =={n}({err * 255:.2f})"
        print(f"{d.name:45} {f.stem:12} {img.mode:6} {img.size[0]}x{img.size[1]} {stats}{same}")
