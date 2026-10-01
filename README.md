# Alcohol references in superimposed image text

Classifies an image as `alcohol` or `non_alcohol` based on the text superimposed on it.
The pipeline is OCR (EasyOCR, CPU only), then text cleaning, then a logistic-regression
classifier over character n-grams and an alcohol word list.

The full write-up (data exploration, design choices, metrics, failure cases, next steps)
is in **[notebooks/report.ipynb](notebooks/report.ipynb)**. This README covers how to run it
and summarises the results.

## Results

Held-out test set: 240 images (20% stratified split, seed 42), never used for model or threshold selection.

| | Precision | Recall | F1 | Accuracy | PR-AUC |
|---|---|---|---|---|---|
| Word-list rule only (no training) | 0.988 | 0.675 | 0.802 | 0.833 | – |
| **Final model** (threshold 0.34) | 0.827 | **0.917** | **0.870** | **0.863** | **0.961** |

Cross-validation on the training split gave F1 0.89 and PR-AUC 0.967. Two small word-list fixes
were made after looking at test errors from a first model (details in the report), so the test
numbers are slightly optimistic. With 240 test images, expect roughly ±4% on accuracy.

## Running predictions on new images

The trained model ships in the repo (`models/model.joblib`), so **no training or dataset is
needed to predict**: set up the environment, then point `predict.py` at a folder of images.

### 1. Set up (once)

Tested with **Python 3.13** on macOS (Apple Silicon), CPU only. No GPU is needed.

macOS / Linux:

```bash
cd DoubleVerify
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
cd DoubleVerify
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Predict

```bash
python predict.py --input-dir /path/to/images
```

This writes **`results.csv`** to the current folder and ends with a summary line:

```
INFO Found 240 images in /path/to/images
OCR: 100%|██████████| 240/240 [09:12<00:00,  2.30s/img]
INFO Wrote 240 predictions (118 alcohol, 122 non_alcohol) to /.../results.csv
```

`results.csv` has a header row and one row per image. `prediction` is always `alcohol`
or `non_alcohol`:

```
file_name,prediction
0000a.jpg,alcohol
0059na.jpg,non_alcohol
```

**What is read:**
* All images under `--input-dir`, **including subfolders**: `.jpg .jpeg .png .bmp .webp .tif .tiff`
  (any capitalisation). Other files are ignored.
* The prediction uses the image pixels only. File names and folder names are never used,
  so a labelled folder layout can be passed in directly.
* A file that can't be opened as an image gets a warning and is predicted `non_alcohol`,
  so `results.csv` still covers every file.

**Options** (`python predict.py --help`):

| Option | Default | Purpose |
|---|---|---|
| `--input-dir DIR` | required | Folder of images to classify |
| `--output PATH` | `results.csv` | Where to write the results |
| `--scores PATH` | none | Also write `file_name,prob_alcohol,prediction`, useful for inspecting borderline cases |
| `--model PATH` | `models/model.joblib` | Use a different trained model |
| `--ocr-cache PATH` | none | Reuse OCR from `scripts/cache_ocr.py` (development only) |

**Timing:** about **2–3 s per image** on a laptop CPU (≈ 10 min per 250 images), because each
image is read in 4–8 orientations. Almost all of that is OCR; the classifier itself takes milliseconds.

**How the decision is made:** the model outputs P(alcohol), and an image is labelled `alcohol`
when P ≥ 0.338. That threshold was chosen by cross-validation and is stored inside the
model file. `--scores` shows the probabilities, if you want to apply a different trade-off
between precision and recall.

### 3. Or call it from Python

```python
from alcohol_clf.predictor import AlcoholPredictor

predictor = AlcoholPredictor()                    # loads models/model.joblib
predictor.predict("photo.jpg")                    # ('alcohol', 0.99995)
predictor.predict_many(["a.jpg", "b.png"])        # [('non_alcohol', 0.0043), ('alcohol', 0.99999)]
predictor.predict(bgr_array)                      # also accepts an image already loaded with cv2
```

Run this from the project folder (or add it to `PYTHONPATH`). EasyOCR is loaded on the first
call and then reused.

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| First run pauses, then shows a download progress bar | EasyOCR downloads its weights once (~95 MB) to `~/.EasyOCR/model/`. Needs internet the first time only. For an offline machine, copy that folder from a machine that already has it. |
| `SSL: CERTIFICATE_VERIFY_FAILED` during the download | Handled automatically via `certifi`. If it persists, run `pip install --upgrade certifi`. |
| `ModuleNotFoundError: No module named 'alcohol_clf'` | Run `predict.py` from the project folder, or with its full path (`python /path/to/DoubleVerify/predict.py …`). |
| `ModuleNotFoundError` for `cv2`, `sklearn`, `easyocr`… | The virtual environment isn't active. Activate it (step 1) and retry. |
| `No images found in …` | The folder has no files with the extensions listed above. |
| Warnings mentioning `torch` or `quantize` | Harmless messages from PyTorch/EasyOCR on CPU. |

### Scoring a results file

```bash
# labels from folder layout <dir>/alcohol/*, <dir>/non_alcohol/*
python scripts/evaluate.py --results results.csv --labels-dir /path/to/labelled_images
# or from a CSV with columns file_name,label
python scripts/evaluate.py --results results.csv --labels-csv truth.csv
```

Prints precision, recall, F1, accuracy and the confusion counts, then lists every misclassified file.

## How it works

```
image ─► EasyOCR in up to 8 orientations (4 flips; +4 more after deskewing tilted text)
      ─► pick the orientation that reads as real words        alcohol_clf/select.py
      ─► clean text: accents, leetspeak (v0dka), broken words   alcohol_clf/text_norm.py
      ─► features: char 2–5-gram TF-IDF + 9 word-list features  alcohol_clf/features.py
                   (fuzzy, tolerant of OCR errors,              alcohol_clf/lexicon.py
                    negation-aware)
      ─► logistic regression, tuned threshold                  alcohol_clf/model.py
```

Key design decisions (reasoning in the report, §2):

* **Why OCR + text rather than an image model:** the label depends on what the caption
  *says*; the photo underneath is unrelated.
* **Why several orientations:** about 38% of captions read best mirrored, upside down or tilted,
  and EasyOCR only reads upright text.
* **Domain word list:** 76 generic alcohol words and 83 brands, matched by word length:
  short words must match exactly ("gin" is not in "ginger"), long words match fuzzily
  ("budweis?r"). It also handles OCR letter confusions (rn↔m, l↔i) and masks negations
  ("non-alcoholic beer", "root beer").
* **Why logistic regression:** it tied with random forest in cross-validation (PR-AUC
  0.9669 vs 0.9673), and it is interpretable, has better-behaved probabilities, and is
  a small file (160 KB).

## Reproducing everything

```bash
python scripts/cache_ocr.py      # OCR all labelled images into cache/ocr_raw.jsonl (~45 min)
python scripts/train.py          # split, 5-fold CV model selection, test, save the model
python -m pytest tests           # regression tests: text cleaning, matching, prediction
jupyter nbconvert --to notebook --execute --inplace notebooks/report.ipynb
```

The labelled data is expected at `images/alcohol/` and `images/non_alcohol/` (set in
`alcohol_clf/config.py`). Everything is seeded (seed 42) and runs on CPU only.
`train.py` writes `reports/split.json`, `cv_results.csv`, `metrics.json` and `test_predictions.csv`.
The shipped `models/model.joblib` is refit on all 1,200 images with the settings chosen by
cross-validation.

To open the report in VS Code or Jupyter, select this project's `.venv` as the kernel.

## Project layout

```
predict.py                 stand-alone inference -> results.csv
alcohol_clf/               library shared by training and inference
  config.py                paths, labels, seed
  ocr.py                   EasyOCR with flips + deskew
  select.py                choose the best OCR orientation
  text_norm.py             text normalisation (leetspeak, accents, squashing)
  lexicon.py               alcohol / non-alcohol vocabulary, fuzzy matcher, negations
  features.py              char n-gram and word-list feature transformers
  model.py                 pipeline definition, save/load, prediction
  predictor.py             AlcoholPredictor: Python API used by predict.py
  metrics.py               metric helpers
  data.py                  image listing, OCR-cache I/O
scripts/
  cache_ocr.py             OCR the labelled dataset once
  train.py                 model selection, test evaluation, save model
  evaluate.py              score a results.csv
tests/test_lexicon.py      regression tests
models/model.joblib        trained model (used by predict.py)
report.ipynb     the report
reports/                   metrics, CV results, test predictions, figures
```

## Known limitations

* **OCR is the ceiling.** About 17% of images give no recognisable words (thin, script or
  low-contrast fonts). Those are close to a coin flip, and most errors come from them.
* **Template artifacts.** Some caption-template words appear in only one class in this
  dataset ("looking for", "enjoy"). The n-gram features partly learn them, which may not
  carry over to differently generated data.
* **Scene text** in the photo is read along with the caption and can cause false hits.
* **Short alcohol words stuck to a neighbour** ("ginismooth") are missed on purpose, to avoid
  matching "ginger".
* **English only.**

The report (§8) lists the next steps, ordered by expected value. The top one is better OCR
for hard captions: a second OCR engine, upscaling and binarising the image, and re-reading
each detected text line separately.
