import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

# ===== Settings =====
QUICK_TEST = False        # True = tiny test run to check everything works. Set False for the real run.
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 5          # phase 1: train only the new final layer
EPOCHS_FINE = 4          # phase 2: gently adjust the top of the base model
SEED = 42

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_NAME = "quick_test.keras" if QUICK_TEST else "chestview_model.keras"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)

tf.keras.utils.set_random_seed(SEED)


# ===== 1. Load the images =====
def load(split, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, split),
        labels="inferred",
        label_mode="binary",
        class_names=["AP", "PA"],     # AP = 0, PA = 1 (the backend expects this order)
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )


train_ds = load("train", True)
val_ds = load("val", False)
test_ds = load("test", False)
print("Class order:", train_ds.class_names, "(AP = 0, PA = 1)")

if QUICK_TEST:
    train_ds = train_ds.take(20)
    val_ds = val_ds.take(5)
    test_ds = test_ds.take(5)
    EPOCHS_HEAD = 1
    EPOCHS_FINE = 1
    print("QUICK TEST MODE: tiny run, the result will NOT be accurate.")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ===== 2. Build the model =====
# The model takes raw pixel values (0 to 255), like the backend sends them.
augment = tf.keras.Sequential(
    [
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomRotation(0.03),
        layers.RandomContrast(0.1),
    ],
    name="augmentation",
)

base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
)
base.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = augment(inputs)                          # only active during training
x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)   # 0..255 -> -1..1
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)   # probability of PA
model = tf.keras.Model(inputs, outputs)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=2, restore_best_weights=True
)

# ===== 3. Phase 1: train the new final layer =====
print("\n=== Phase 1: training the final layer ===")
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
hist1 = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=[early_stop]
)

# ===== 4. Phase 2: fine-tune the top of the base model =====
print("\n=== Phase 2: fine-tuning ===")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
hist2 = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS_FINE, callbacks=[early_stop]
)

# ===== 5. Save the model =====
model.save(MODEL_PATH)
print("\nModel saved to:", MODEL_PATH)

# ===== 6. Save training curves =====
def joined(key):
    return hist1.history[key] + hist2.history[key]


fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(joined("accuracy"), label="train")
axes[0].plot(joined("val_accuracy"), label="validation")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].plot(joined("loss"), label="train")
axes[1].plot(joined("val_loss"), label="validation")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()
plt.tight_layout()
curves_name = "quick_test_curves.png" if QUICK_TEST else "training_curves.png"
plt.savefig(os.path.join(MODELS_DIR, curves_name))
print("Training curves saved in the models folder.")

# ===== 7. Quick check on the test images =====
print("\n=== Test set result (images the model has never seen) ===")
loss, acc = model.evaluate(test_ds, verbose=0)
print("Test accuracy: %.1f%%" % (acc * 100))