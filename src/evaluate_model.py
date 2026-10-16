import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(PROJECT_DIR, "data", "processed", "test")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "chestview_model.keras")

# ----- Load the model and the test images -----
model = tf.keras.models.load_model(MODEL_PATH)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["AP", "PA"],     # AP = 0, PA = 1
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,                # keep the order so labels line up with predictions
)

# ----- Predict -----
y_true = np.concatenate([y.numpy().ravel() for _, y in test_ds]).astype(int)
probs = model.predict(test_ds, verbose=0).ravel()    # probability of PA
y_pred = (probs >= 0.5).astype(int)

# ----- Measure -----
accuracy = (y_true == y_pred).mean()
report = classification_report(y_true, y_pred, target_names=["AP", "PA"], digits=3)
cm = confusion_matrix(y_true, y_pred)
auc = roc_auc_score(y_true, probs)

text = (
    "Test images: %d\n" % len(y_true)
    + "Accuracy: %.2f%%\n" % (accuracy * 100)
    + "ROC AUC: %.4f\n\n" % auc
    + report
    + "\nConfusion matrix (rows = actual, columns = predicted; order AP, PA):\n"
    + str(cm)
    + "\n"
)
print(text)

with open(os.path.join(MODELS_DIR, "evaluation_report.txt"), "w") as f:
    f.write(text)

# ----- Draw the confusion matrix -----
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_xticklabels(["AP", "PA"])
ax.set_yticks([0, 1])
ax.set_yticklabels(["AP", "PA"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion matrix (test set)")
for i in range(2):
    for j in range(2):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "confusion_matrix.png"))
print("Saved confusion_matrix.png and evaluation_report.txt in the models folder.")