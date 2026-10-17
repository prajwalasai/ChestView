import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

# ===== Settings =====
# Folder that holds your mixed images (subfolders are fine). Keep the r before the quotes.
MIXED_FOLDER = r"C:\Users\mache\Downloads\images\images"
MAX_IMAGES = 200
IMG_SIZE = (224, 224)
SEED = 42

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "chestview_model.keras")
LABELS_CSV = os.path.join(PROJECT_DIR, "data", "labels.csv")
SUBSET_CSV = os.path.join(PROJECT_DIR, "data", "processed", "subset_labels.csv")

model = tf.keras.models.load_model(MODEL_PATH)


def find_col(df, keyword):
    for c in df.columns:
        cleaned = c.lower().replace(" ", "").replace("_", "").replace("-", "")
        if keyword in cleaned:
            return c
    return None


def predict(path):
    # same steps as backend/app.py
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img, dtype="float32"), axis=0)
    p = float(model.predict(arr, verbose=0)[0][0])      # probability of PA
    return ("PA" if p >= 0.5 else "AP"), p


# ----- Look-up tables from the CSV files -----
labels = pd.read_csv(LABELS_CSV)
lv = find_col(labels, "viewposition")
li = find_col(labels, "imageindex")
true_view = dict(zip(labels[li].astype(str), labels[lv].astype(str).str.strip().str.upper()))

subset = pd.read_csv(SUBSET_CSV)
split_of = dict(zip(subset[find_col(subset, "imageindex")].astype(str), subset["split"]))

# ----- Collect the images -----
if not os.path.isdir(MIXED_FOLDER):
    raise SystemExit("Folder not found: " + MIXED_FOLDER)

files = []
for folder, _, names in os.walk(MIXED_FOLDER):
    for name in names:
        if name.lower().endswith((".png", ".jpg", ".jpeg")):
            files.append(os.path.join(folder, name))

print("Images found in folder:", len(files))
random.seed(SEED)
if len(files) > MAX_IMAGES:
    files = random.sample(files, MAX_IMAGES)

# ----- Predict -----
rows = []
for path in files:
    name = os.path.basename(path)
    pred, p = predict(path)
    rows.append((
        name,
        true_view.get(name, "unknown"),
        pred,
        round(p, 3),
        split_of.get(name, "not in subset"),
    ))

res = pd.DataFrame(rows, columns=["image", "true", "pred", "p_pa", "split"])
known = res[res["true"].isin(["AP", "PA"])].copy()
unknown = res[~res["true"].isin(["AP", "PA"])]

print("\nImages checked:", len(res))
print("  label found in labels.csv:", len(known))
print("  not in labels.csv (label unknown):", len(unknown))

if len(known):
    known["correct"] = known["true"] == known["pred"]
    print("\n=== Accuracy by true view ===")
    for view in ["AP", "PA"]:
        sub = known[known["true"] == view]
        if len(sub):
            print("%s images: %d of %d correct (%.1f%%)"
                  % (view, sub["correct"].sum(), len(sub), 100 * sub["correct"].mean()))

    print("\n=== Correct / total, by where the image was used ===")
    print(known.groupby(["split", "true"])["correct"].agg(["sum", "count"]))

    wrong = known[~known["correct"]]
    print("\n=== Examples of mistakes (up to 10) ===")
    print(wrong.head(10).to_string(index=False))

if len(unknown):
    print("\n=== Images with unknown label: what the model says ===")
    print(unknown["pred"].value_counts())
    print(unknown.head(10).to_string(index=False))