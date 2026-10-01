"""Programmatic inference API.

    from alcohol_clf.predictor import AlcoholPredictor

    predictor = AlcoholPredictor()                   # loads models/model.joblib
    predictor.predict("photo.jpg")                   # -> ("alcohol", 0.97)
    predictor.predict_many(["a.jpg", "b.png"])       # -> [("alcohol", 0.97), ...]

`predict.py` is a thin command-line wrapper around this class.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .model import MODEL_PATH, load_model, predict_proba, to_labels


class AlcoholPredictor:
    """Image -> ("alcohol" | "non_alcohol", P(alcohol)).

    EasyOCR is loaded lazily on the first image that needs OCR (it takes a few
    seconds), so constructing the predictor is cheap.
    """

    def __init__(self, model_path: Path = MODEL_PATH, ocr_cache: dict | None = None):
        self.bundle = load_model(model_path)
        self.threshold = float(self.bundle["threshold"])
        self.ocr_cache = ocr_cache or {}  # {file_name: {"ocr": [...]}} from cache_ocr.py
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            from .ocr import OCREngine
            self._engine = OCREngine()
        return self._engine

    def read_text(self, image: str | Path | np.ndarray) -> list[dict]:
        """Raw OCR output for one image (path or BGR array), all orientations."""
        if isinstance(image, np.ndarray):
            return self.engine.read(image)
        path = Path(image)
        if path.name in self.ocr_cache:
            return self.ocr_cache[path.name]["ocr"]
        return self.engine.read_path(path)

    def predict_many(self, images) -> list[tuple[str, float]]:
        ocr_list = [self.read_text(im) for im in images]
        return self.predict_ocr(ocr_list)

    def predict(self, image) -> tuple[str, float]:
        return self.predict_many([image])[0]

    def predict_ocr(self, ocr_list: list[list[dict] | None]) -> list[tuple[str, float]]:
        """Classify already-extracted OCR output (one entry per image).

        An entry of None marks an image that could not be read at all; it is
        predicted non_alcohol with P=0 rather than scored by the model, which
        has never seen a missing image.
        """
        ok = [i for i, ocr in enumerate(ocr_list) if ocr is not None]
        probs = np.zeros(len(ocr_list))
        if ok:
            probs[ok] = predict_proba(self.bundle, [ocr_list[i] for i in ok])
        return list(zip(to_labels(probs, self.threshold), map(float, probs)))
