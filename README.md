# ChestView

**Automated chest X-ray view position classification using deep learning.**

ChestView looks at a chest X-ray image and predicts whether it is a **PA (Posterior-Anterior)** or **AP (Anterior-Posterior)** view. It does **not** diagnose any disease. It only identifies how the X-ray was taken.

## How it works

```
Chest X-ray -> Image preprocessing -> Deep learning model -> PA / AP prediction
```

1. The user uploads a chest X-ray (JPG or PNG) on the website.
2. The Flask backend resizes it to 224x224 and passes it to the trained model.
3. The model returns the probability of PA, and the website shows the predicted view with a confidence bar.

## Features

- Drag-and-drop image upload with preview
- Predicted view (PA or AP) with confidence score
- Input validation (file type and size)
- Clean, responsive web interface

## Tech stack

| Part | Technology |
|---|---|
| Model | TensorFlow / Keras, MobileNetV2 (transfer learning) |
| Backend | Python, Flask |
| Frontend | HTML, CSS, JavaScript |
| Data handling | pandas, NumPy, Pillow, scikit-learn |

## Project structure

```
ChestView/
|-- backend/
|   `-- app.py                 Flask server and /predict API
|-- frontend/
|   |-- index.html             Web page
|   |-- style.css              Styling
|   `-- script.js              Upload and result logic
|-- src/
|   |-- inspect_data.py        Checks the dataset and labels
|   |-- prepare_data.py        Builds the balanced, resized dataset
|   |-- train_model.py         Trains and saves the model
|   |-- evaluate_model.py      Evaluates the model on test images
|   `-- check_mixed.py         Checks the model on a folder of images
|-- models/
|   |-- chestview_model.keras  Trained model
|   |-- training_curves.png
|   |-- confusion_matrix.png
|   `-- evaluation_report.txt
|-- data/                      Dataset (not included in the repository)
|-- requirements.txt
`-- README.md
```

## Dataset

- Chest X-ray images with a labels CSV (columns include Image Index, Patient ID and View Position).
- 14,999 images in total: 9,882 PA and 5,117 AP.
- Source: (add your dataset name and link here)
- The dataset is not included in this repository because of its size.

### Preparation

- Only the **View Position** column is used as the label (PA or AP).
- Images are split **by patient** into train (70%), validation (15%) and test (15%), so no patient appears in more than one split.
- Each split is **balanced** (equal numbers of PA and AP images).
- All images are resized to 224x224. The final set has 10,234 images.

## Model

- **MobileNetV2** pretrained on ImageNet, with a new final layer that outputs the probability of PA.
- Light data augmentation (small zoom, shift, rotation and contrast changes).
- Phase 1: train the new final layer. Phase 2: fine-tune the top layers of MobileNetV2 with a low learning rate.

## Results (test set, 1,560 unseen images from the same dataset)

| Metric | Value |
|---|---|
| Accuracy | 95.51% |
| ROC AUC | 0.9932 |
| AP precision / recall | 0.931 / 0.983 |
| PA precision / recall | 0.982 / 0.927 |

Confusion matrix (rows = actual, columns = predicted):

| | Predicted AP | Predicted PA |
|---|---|---|
| **Actual AP** | 767 | 13 |
| **Actual PA** | 57 | 723 |

On this test set the model is slightly more likely to label a PA image as AP than the other way around.

## Getting started (Windows)

Requires Python 3.10, 3.11 or 3.12.

```
git clone https://github.com/prajwalasai/ChestView.git
cd ChestView
py -3.11 -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Run the website

The trained model is included in `models/`, so you can run the website right away:

```
python backend/app.py
```

Then open **http://127.0.0.1:5000/** in your browser.

### Retrain the model (optional)

1. Put the labels file at `data/labels.csv` and extract the images somewhere on your computer.
2. Set `IMAGES_ROOT` at the top of `src/inspect_data.py` and `src/prepare_data.py` to your images folder.
3. Run:

```
python src/inspect_data.py
python src/prepare_data.py
python src/train_model.py
python src/evaluate_model.py
```

Set `QUICK_TEST = False` in `src/train_model.py` for the full training run. Training takes roughly 15 to 30 minutes on a laptop CPU.

## Limitations

- The model only classifies view position (PA or AP). It cannot detect diseases.
- The reported accuracy applies to images from the same dataset the model was trained on.
- The model was trained on a single dataset. In informal testing with a few images downloaded from the internet, some images believed to be AP were classified as PA. Internet images often differ from the training data (compression, cropping, shape, contrast), and their view labels could not always be verified, so performance on images from other sources has not been measured.
- AP images are often taken at the bedside and may contain devices such as tubes or monitors, so the model may partly rely on such cues.
- For educational use only. It is not a medical device.

## Team

- Macherla Prajwala Sai
- teammate's name-Motla Roja
                 -Mugdha Arelly
                 -Sanith Kumar
                 -D.Sreethu