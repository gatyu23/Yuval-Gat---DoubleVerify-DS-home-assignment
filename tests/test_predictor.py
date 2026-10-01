"""Tests for the shipped model's prediction path (no OCR or dataset needed).

OCR output is built by hand in the format `OCREngine.read` returns, so these
run in well under a second and check the saved model end to end.
"""
from alcohol_clf.predictor import AlcoholPredictor


def ocr(*texts, conf=0.9):
    """One upright OCR variant containing the given text lines."""
    box = [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]]
    return [{"variant": "orig",
             "dets": [{"text": t, "conf": conf, "box": box} for t in texts]}]


def test_predictions_and_output_format():
    predictor = AlcoholPredictor()
    results = predictor.predict_ocr([
        ocr("Don't miss limited edition nightlife bourbon event"),
        ocr("Check out iced herbal tea special"),
        None,  # a file that could not be read as an image
    ])
    labels = [label for label, _ in results]
    assert labels == ["alcohol", "non_alcohol", "non_alcohol"]
    assert results[2][1] == 0.0
    assert all(0.0 <= p <= 1.0 for _, p in results)
    assert all(label in {"alcohol", "non_alcohol"} for label in labels)


def test_negated_alcohol_is_not_flagged():
    label, _ = AlcoholPredictor().predict_ocr([ocr("Try our non-alcoholic beer")])[0]
    assert label == "non_alcohol"
