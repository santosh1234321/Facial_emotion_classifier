import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Config ─────────────────────────────────────────────────────────────────────
train_dir  = "dataset/train"
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
AUTOTUNE   = tf.data.AUTOTUNE

# ── Load Data ──────────────────────────────────────────────────────────────────
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir, validation_split=0.2, subset="training",
    seed=42, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode='categorical'
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir, validation_split=0.2, subset="validation",
    seed=42, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode='categorical'
)

class_names = train_ds.class_names
n_classes   = len(class_names)
print("Classes:", class_names)

# ── Fast Class Weight Computation (count files directly) ───────────────────────
label_counts = np.array([
    len(os.listdir(os.path.join(train_dir, cls)))
    for cls in class_names
])
total = label_counts.sum()
class_weights = {i: total / (n_classes * c) for i, c in enumerate(label_counts)}

# Boost harder classes
class_weights[0] *= 1.5   # Angry
class_weights[2] *= 1.5   # Fear
class_weights[3] *= 2.0   # Happy
class_weights[4] *= 1.8   # Neutral
print("Class weights:", class_weights)

# ── Augmentation ───────────────────────────────────────────────────────────────
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.3),
    layers.RandomBrightness(0.3),
    layers.RandomTranslation(0.15, 0.15),
], name="augmentation")

train_ds = (train_ds
    .map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE))
val_ds = val_ds.prefetch(AUTOTUNE)

# ── Model: MobileNetV2 ─────────────────────────────────────────────────────────
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights='imagenet'
)
base_model.trainable = False

inputs  = tf.keras.Input(shape=(224, 224, 3))
x       = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x       = base_model(x, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dense(256, activation='relu')(x)
x       = layers.Dropout(0.4)(x)
x       = layers.Dense(128, activation='relu')(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(n_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ── Callbacks ──────────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "model/best_model.h5", monitor='val_accuracy',
        save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=8,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.3,
        patience=4, min_lr=1e-8, verbose=1
    )
]

# ── Phase 1: Frozen base ───────────────────────────────────────────────────────
print("\n===== Phase 1: Frozen base =====")
model.fit(train_ds, validation_data=val_ds,
          epochs=25, class_weight=class_weights, callbacks=callbacks)

# ── Phase 2: Fine-tune ─────────────────────────────────────────────────────────
print("\n===== Phase 2: Fine-tuning =====")
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(5e-6),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_ds, validation_data=val_ds,
          epochs=20, class_weight=class_weights, callbacks=callbacks)

# ── Save ───────────────────────────────────────────────────────────────────────
model.save("model/emotion_model.h5")
print("Training complete! Model saved to model/emotion_model.h5")
