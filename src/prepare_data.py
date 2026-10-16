import os

import numpy as np
import pandas as pd
from PIL import Image

# ===== Settings =====
IMAGES_ROOT = r"C:\Users\mache\Downloads\images"   # your extracted images folder
IMG_SIZE = (224, 224)
SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15          # the remaining 15% becomes the test set

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_DIR, "data", "labels.csv")
OUT_DIR = os.path.join(PROJECT_DIR, "data", "processed")


def find_col(df, keyword):
    for c in df.columns:
        cleaned = c.lower().replace(" ", "").replace("_", "").replace("-", "")
        if keyword in cleaned:
            return c
    return None


# ----- 1. Read the CSV -----
df = pd.read_csv(CSV_PATH)
view_col = find_col(df, "viewposition")
img_col = find_col(df, "imageindex")
pid_col = find_col(df, "patientid")
print("Columns used:", view_col, "|", img_col, "|", pid_col)

df[view_col] = df[view_col].astype(str).str.strip().str.upper()
df = df[df[view_col].isin(["PA", "AP"])].copy()
print("Rows with PA or AP:", len(df))

# ----- 2. Split by patient -----
patients = df[pid_col].unique()
rng = np.random.RandomState(SEED)
rng.shuffle(patients)

n = len(patients)
train_end = int(n * TRAIN_FRAC)
val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
train_p = set(patients[:train_end])
val_p = set(patients[train_end:val_end])


def which_split(pid):
    if pid in train_p:
        return "train"
    if pid in val_p:
        return "val"
    return "test"


df["split"] = df[pid_col].apply(which_split)

# ----- 3. Balance PA and AP inside each split -----
parts = []
for split in ["train", "val", "test"]:
    sub = df[df["split"] == split]
    ap = sub[sub[view_col] == "AP"]
    pa = sub[sub[view_col] == "PA"]
    k = min(len(ap), len(pa))
    parts.append(ap.sample(k, random_state=SEED))
    parts.append(pa.sample(k, random_state=SEED))

subset = pd.concat(parts).reset_index(drop=True)
print("\nImages selected (balanced):")
print(subset.groupby(["split", view_col]).size())

# ----- 4. Find the image files on disk -----
found = {}
for folder, _, files in os.walk(IMAGES_ROOT):
    for name in files:
        if name.lower().endswith((".png", ".jpg", ".jpeg")):
            found[name] = os.path.join(folder, name)
print("\nImage files found on disk:", len(found))

# ----- 5. Resize and save -----
print("\nResizing images... this takes several minutes. Please wait.")
done = 0
skipped = 0
total = len(subset)

for _, row in subset.iterrows():
    name = str(row[img_col])
    src_path = found.get(name)
    if src_path is None:
        skipped += 1
        continue

    out_folder = os.path.join(OUT_DIR, row["split"], row[view_col])
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, os.path.splitext(name)[0] + ".jpg")

    if not os.path.exists(out_path):          # lets you re-run without redoing work
        try:
            with Image.open(src_path) as im:
                im = im.convert("RGB").resize(IMG_SIZE)
                im.save(out_path, "JPEG", quality=95)
        except Exception as e:
            print("Skipped", name, "->", e)
            skipped += 1
            continue

    done += 1
    if done % 500 == 0:
        print(done, "of", total, "done")

subset.to_csv(os.path.join(OUT_DIR, "subset_labels.csv"), index=False)

print("\n=== Finished ===")
print("Saved:", done, "| Skipped:", skipped)
print("Output folder:", OUT_DIR)