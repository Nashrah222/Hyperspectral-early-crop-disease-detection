import os

# ================================================================
# MUST be set BEFORE importing TensorFlow — fixes CUDA_ERROR_NOT_PERMITTED
# ================================================================
os.environ.setdefault('TF_GPU_ALLOCATOR',          'cuda_malloc_async')
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
os.environ.setdefault('CUDA_VISIBLE_DEVICES',      '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL',      '2')
# ================================================================

import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling1D, Reshape,
    Multiply, Input
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import os

# =================================================================
# GPU CHECK
# =================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU DETECTED: {len(gpus)} GPU(s) available.")
    except RuntimeError as e:
        print(e)
else:
    print("❌ NO GPU DETECTED — using CPU.")
# =================================================================


def squeeze_excitation_block(inputs, ratio=8):
    """
    Channel Attention Block.
    Learns which of the 512 feature maps matter most for disease detection
    and amplifies those while suppressing noisy ones.
    """
    filters = inputs.shape[-1]
    se = GlobalAveragePooling1D()(inputs)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, filters))(se)
    return Multiply()([inputs, se])


def build_model(input_shape, num_classes):
    """
    Improved CNN with SE attention block.
    - input_shape: (Bands, 3)  — 3 channels: raw, 1st deriv, 2nd deriv
    - Dropout reduced to 0.3 (was 0.6)
    - SE attention after final conv layer
    - Two dense layers instead of one
    """
    inp = Input(shape=input_shape)

    # Conv Block 1
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Conv Block 2
    x = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Conv Block 3
    x = Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Conv Block 4
    x = Conv1D(512, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # SE Attention — learns which spectral features matter most
    x = squeeze_excitation_block(x, ratio=8)

    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)

    # Classification Head
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inp, outputs=out)


def train_model():
    print("=========================================================")
    print("   ⚡ IMPROVED CNN — SE ATTENTION + TRIPLE CHANNEL ⚡     ")
    print("=========================================================")

    # ── Load ──────────────────────────────────────────────────────
    print("\n[STEP 1] Loading data...")
    if not os.path.exists('data/X_cleaned.npy'):
        print("❌ data/X_cleaned.npy not found. Run noise_removal.py first.")
        return

    X = np.load('data/X_cleaned.npy')   # shape: (N_pixels, Bands, 3)
    y = np.load('data/Y_labels.npy')    # shape: (N_pixels,)  dtype: float32

    print(f"   X shape : {X.shape}")
    print(f"   Labels  : {np.unique(y)}  (dtype: {y.dtype})")
    print(f"   Counts  : {dict(zip(*np.unique(y, return_counts=True)))}")

    # ── Remove background pixels (label == 0) ─────────────────────
    # Background = soil, shadow, pot edges — not plant tissue.
    # Training on them wastes capacity and lowers accuracy.
    print("\n[STEP 2] Removing background pixels (label == 0.0)...")
    mask = y > 0
    X = X[mask]
    y = y[mask]
    print(f"   Kept    : {mask.sum():,} plant pixels")
    print(f"   Removed : {(~mask).sum():,} background pixels")
    print(f"   Remaining labels: {np.unique(y)}")

    # ── Build and SAVE label map ───────────────────────────────────
    # Labels are float32 (0.0, 1.0, 2.0, 3.0) — handle as floats.
    # We save this map so evaluate.py and region_props.py use the
    # EXACT same mapping and don't produce mismatched results.
    unique_classes = np.unique(y)                             # e.g. [1.0, 2.0, 3.0]
    label_map    = {float(old): int(new) for new, old in enumerate(unique_classes)}
    reverse_map  = {int(new): float(old) for new, old in enumerate(unique_classes)}
    num_classes  = len(unique_classes)

    os.makedirs('models', exist_ok=True)
    with open('models/label_map.json', 'w') as f:
        json.dump({
            'label_map'     : {str(k): v for k, v in label_map.items()},
            'reverse_map'   : {str(k): v for k, v in reverse_map.items()},
            'unique_classes': [float(c) for c in unique_classes],
            'num_classes'   : num_classes,
            'background_label': 0.0
        }, f, indent=2)

    print(f"\n   ✔ Label map saved → models/label_map.json")
    print(f"   raw → model index : {label_map}")
    print(f"   model index → raw : {reverse_map}")

    # Apply remapping
    y = np.array([label_map[float(lbl)] for lbl in y])
    # y is now clean integers: 0, 1, 2

    # ── Class weights ─────────────────────────────────────────────
    # Disease pixels are fewer than healthy — weights compensate for this
    print("\n[STEP 3] Computing class weights...")
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    class_weights_dict = dict(enumerate(weights))
    print(f"   Class weights: {class_weights_dict}")

    # ── Train / test split ────────────────────────────────────────
    y_encoded = to_categorical(y, num_classes=num_classes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"\n   Train : {X_train.shape[0]:,} samples")
    print(f"   Test  : {X_test.shape[0]:,} samples")

    # ── tf.data pipeline ──────────────────────────────────────────
    BATCH_SIZE = 64
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(buffer_size=8000)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ── Build model ───────────────────────────────────────────────
    print("\n[STEP 4] Building model...")
    input_shape = (X.shape[1], X.shape[2])   # (Bands, 3)
    model = build_model(input_shape, num_classes)
    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    # ── Callbacks ─────────────────────────────────────────────────
    os.makedirs('models/saved_models', exist_ok=True)
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3,
            min_lr=1e-6, verbose=1
        ),
        EarlyStopping(
            monitor='val_loss', patience=12,
            restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            filepath='models/saved_models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
    ]

    # ── Train (single block — duplicate removed) ──────────────────
    print(f"\n[STEP 5] Training...")
    history = model.fit(
        train_dataset,
        epochs=80,
        validation_data=test_dataset,
        class_weight=class_weights_dict,
        callbacks=callbacks
    )

    model.save('models/saved_models/final_model.h5')

    best_val = max(history.history['val_accuracy'])
    print(f"\n✔ Training complete!")
    print(f"   Best validation accuracy : {best_val*100:.1f}%")
    print(f"   Label map saved to       : models/label_map.json")
    print(f"   Best model saved to      : models/saved_models/best_model.h5")
    print("=========================================================")

    return history


if __name__ == "__main__":
    train_model()