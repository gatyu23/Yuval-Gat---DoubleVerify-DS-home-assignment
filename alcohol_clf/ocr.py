"""OCR stage: EasyOCR with geometric test-time augmentation (TTA).

The dataset contains text that is mirrored, upside down, rotated, or a mix of
these. EasyOCR only reads roughly upright, left-to-right text, so we run it on
several geometric variants of each image and keep all raw outputs:

  * flips: orig / hflip (mirrored) / vflip (mirrored + upside down) / rot180
  * deskew: if the text is tilted, rotate it level first, then apply the
    same four flips. The tilt angle is estimated from the free-form
    quadrilaterals of EasyOCR's CRAFT detector, which is cheap compared to a
    brute-force search over angles.

Choosing which variant to trust is a separate step (`select.py`), which lets
us tune that logic offline against cached OCR results.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

# python.org builds of Python on macOS ship without a CA bundle, which breaks
# EasyOCR's first-run model download. Point SSL at certifi's bundle.
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:  # pragma: no cover
    pass

FLIPS = {
    "orig": lambda im: im,
    "hflip": lambda im: cv2.flip(im, 1),
    "vflip": lambda im: cv2.flip(im, 0),
    "rot180": lambda im: cv2.rotate(im, cv2.ROTATE_180),
}
MIN_SKEW_DEG = 5.0  # below this, EasyOCR copes fine without deskewing
PROBE_ANGLES = (25.0, -25.0)  # CRAFT often finds nothing on steeply tilted text


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate counter-clockwise by `angle` degrees, expanding the canvas."""
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(
        image, m, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255)
    )


class OCREngine:
    def __init__(self):
        import easyocr
        import torch

        torch.manual_seed(0)
        self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def _text_angle(self, image: np.ndarray) -> float | None:
        """Length-weighted mean tilt of CRAFT's rotated boxes, in degrees."""
        _, free = self.reader.detect(image)
        angles, weights = [], []
        for poly in free[0]:
            p = np.asarray(poly, dtype=float)
            dx, dy = p[1] - p[0]
            angles.append(np.degrees(np.arctan2(dy, dx)))
            weights.append(np.hypot(dx, dy))
        return float(np.average(angles, weights=weights)) if angles else None

    def estimate_skew(self, image: np.ndarray) -> float | None:
        """Angle that levels the text, or None if it is already ~horizontal."""
        angle = self._text_angle(image)
        if angle is None:
            horizontal, _ = self.reader.detect(image)
            if horizontal[0]:
                return None  # upright text found; nothing to fix
            for probe in PROBE_ANGLES:
                residual = self._text_angle(rotate(image, probe))
                if residual is not None:
                    angle = probe + residual
                    break
                horizontal, _ = self.reader.detect(rotate(image, probe))
                if horizontal[0]:
                    angle = probe
                    break
        if angle is None or abs(angle) < MIN_SKEW_DEG:
            return None
        return angle

    def _read_variant(self, name: str, image: np.ndarray) -> dict:
        dets = self.reader.readtext(image)
        return {
            "variant": name,
            "dets": [
                {
                    "text": text,
                    "conf": float(conf),
                    "box": [[float(x), float(y)] for x, y in box],
                }
                for box, text, conf in dets
            ],
        }

    def read(
        self, image: np.ndarray, flips: bool = True, deskew: bool = True
    ) -> list[dict]:
        """Return one detection list per variant: {"text", "conf", "box"}.

        The flags let a cache built by an older version be extended with only
        the missing stage.
        """
        out = []
        if flips:
            out += [self._read_variant(name, fn(image)) for name, fn in FLIPS.items()]
        if not deskew:
            return out
        angle = self.estimate_skew(image)
        if angle is not None:
            level = rotate(image, angle)
            out += [
                self._read_variant(f"deskew_{name}", fn(level)) | {"angle": angle}
                for name, fn in FLIPS.items()
            ]
        return out

    def read_path(self, path, **kwargs) -> list[dict]:
        # cv2.imread cannot open non-ASCII paths on Windows; decoding from a
        # byte buffer works everywhere.
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        return self.read(image, **kwargs)
