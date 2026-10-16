import os

import pandas as pd
from PIL import Image

# ===========================================================
# EDIT THIS ONE LINE: paste the folder that contains your images.
# Subfolders are fine. Keep the r before the quotes.
IMAGES_ROOT = r"C:\Users\mache\Downloads\images.zip\images"
# ===========================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_DIR, "data", "labels.csv")


def find_col(df, keyword):
    """Finds a column whose name contains the keyword (ignores spaces/case)."""
    for c in df.columns:
        cleaned = c.lower().replace(" ", "").replace("_", "").replace("-", "")
        if keyword in cleaned:
            return c
    return None


# ----- 1. Read the CSV -----
if not os.path.exists(CSV_PATH):
    raise SystemExit("labels.csv not found at: " + CSV_PATH)

df = pd.read_csv(CSV_PATH)
print("=== CSV ===")
print("Rows:", len(df))
print("Columns:", list(df.columns))

view_col = find_col(df, "viewposition")
img_col = find_col(df, "imageindex")
print("\nView column :", view_col)
print("Image column:", img_col)

if view_col is None or img_col is None:
    raise SystemExit("Could not find the view position or image index column.")

# ----- 2. Count PA and AP -----
print("\n=== View position counts ===")
print(df[view_col].value_counts(dropna=False))

# ----- 3. Find the image files on disk -----
print("\n=== Images on disk ===")
if not os.path.isdir(IMAGES_ROOT):
    raise SystemExit("IMAGES_ROOT folder not found: " + IMAGES_ROOT)

found = {}
for folder, _, files in os.walk(IMAGES_ROOT):
    for name in files:
        if name.lower().endswith((".png", ".jpg", ".jpeg")):
            found[name] = os.path.join(folder, name)

print("Image files found:", len(found))

csv_names = set(df[img_col].astype(str))
matched = [n for n in csv_names if n in found]
print("CSV rows that have an image on disk:", len(matched), "of", len(csv_names))

# ----- 4. Look at one image -----
if matched:
    sample = found[matched[0]]
    with Image.open(sample) as im:
        print("\nSample image:", sample)
        print("Size:", im.size, "| Mode:", im.mode)
else:
    print("\nNo CSV image names matched files on disk. Check IMAGES_ROOT.")