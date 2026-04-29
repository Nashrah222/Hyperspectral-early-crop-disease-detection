# ================================================================
# IMPORTANT: These environment variables MUST be set before
# importing TensorFlow. Do not move them below the imports.
# They fix CUDA_ERROR_NOT_PERMITTED on RTX 40-series laptops.
# ================================================================
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES']      = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']      = '2'
os.environ['TF_CUDNN_USE_AUTOTUNE']     = '0'

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cuda_paths = [
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin',
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp',
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin',
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp',
    _base_dir,
    os.path.join(_base_dir, 'gpu_env/Scripts'),
]
os.environ['PATH'] = ';'.join(_cuda_paths) + ';' + os.environ.get('PATH', '')
# ================================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv3D, Conv2D, Flatten, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D,
    Reshape, Multiply, Input
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
)
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[OK] GPU DETECTED: {len(gpus)} GPU(s) - memory growth enabled.")
    except RuntimeError as e:
        print(e)
else:
    print("[!!] No GPU detected - using CPU.")
# ================================================================


def squeeze_excitation_block(inputs, ratio=8):
    filters = inputs.shape[-1]
    se = GlobalAveragePooling2D()(inputs)
    se = Dense(max(filters // ratio, 1), activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, 1, filters))(se)
    return Multiply()([inputs, se])


def build_hybridsn(patch_size, n_bands, num_classes):
    inp = Input(shape=(patch_size, patch_size, n_bands, 1))

    # 3D conv — spectral + spatial together
    x = Conv3D(8,  kernel_size=(3, 3, 7), activation='relu', padding='same')(inp)
    x = BatchNormalization()(x)
    x = Conv3D(16, kernel_size=(3, 3, 5), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    # Reshape 3D -> 2D
    _, h, w, b, f = x.shape
    x = Reshape((h, w, b * f))(x)

    # 2D conv — spatial patterns
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = squeeze_excitation_block(x, ratio=8)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    # Classifier
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inp, outputs=out)


def make_dataset(X, y, batch_size, shuffle=False):
    """Streams batches from CPU RAM to GPU one at a time."""
    n       = len(X)
    indices = np.arange(n)

    def generator():
        idx = indices.copy()
        if shuffle:
            np.random.shuffle(idx)
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            x_batch   = X[batch_idx].astype(np.float32)
            y_batch   = y[batch_idx]
            if shuffle:
                if np.random.rand() > 0.5:
                    x_batch = x_batch[:, :, ::-1, :, :]
                if np.random.rand() > 0.5:
                    x_batch = x_batch[:, ::-1, :, :, :]
                k = np.random.randint(0, 4)
                if k > 0:
                    x_batch = np.rot90(x_batch, k=k, axes=(1, 2))
                x_batch *= np.random.uniform(0.95, 1.05)
                x_batch += np.random.normal(0, 0.005,
                                            x_batch.shape).astype(np.float32)
            yield x_batch, y_batch

    patch_size = X.shape[1]
    n_bands    = X.shape[3]
    num_cls    = y.shape[1]

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, patch_size, patch_size, n_bands, 1),
                          dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_cls), dtype=tf.float32),
        )
    ).repeat().prefetch(tf.data.AUTOTUNE)

    return ds


def plot_training_curves(history, models_dir):
    """Plot and save accuracy + loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'],     label='Train Acc')
    ax1.plot(history.history['val_accuracy'], label='Val Acc')
    ax1.set_title('Accuracy\n(val >> train from epoch 1 = suspect leakage)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['loss'],     label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'training_curves.png'))
    plt.close()
    print(f"   ✔ Saved -> models/training_curves.png")


def plot_confusion_matrix(y_true, y_pred, models_dir):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix\n(one dominant column = always predicts same class)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'confusion_matrix.png'))
    plt.close()
    print(f"   ✔ Saved -> models/confusion_matrix.png")


def run_verification(model, test_ds, val_steps, history, models_dir):
    """
    Full post-training verification:
      - Per-class classification report (printed + saved)
      - Confusion matrix PNG
      - Class distribution
      - Automatic leakage / imbalance diagnosis
    """
    print("\n" + "=" * 57)
    print("   [VERIFY] POST-TRAINING ACCURACY CHECKS")
    print("=" * 57)

    print("\n   Collecting predictions (batch by batch)...")
    y_true_all = []
    y_pred_all = []

    for x_batch, y_batch in test_ds.take(val_steps):
        preds = model.predict_on_batch(x_batch)
        y_true_all.extend(np.argmax(y_batch.numpy(), axis=1))
        y_pred_all.extend(np.argmax(preds, axis=1))

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # Per-class report
    report = classification_report(y_true_all, y_pred_all, digits=3)
    print("\n[VERIFY] Per-class accuracy report:")
    print(report)

    report_path = os.path.join(models_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Best val accuracy: {max(history.history['val_accuracy'])*100:.1f}%\n\n")
        f.write(report)
    print(f"   ✔ Saved -> models/classification_report.txt")

    # Confusion matrix
    plot_confusion_matrix(y_true_all, y_pred_all, models_dir)

    # Class distribution
    print("\n[VERIFY] Test set class distribution:")
    unique, counts = np.unique(y_true_all, return_counts=True)
    dominant_pct = 0.0
    for cls, cnt in zip(unique, counts):
        pct = cnt / len(y_true_all) * 100
        dominant_pct = max(dominant_pct, pct)
        print(f"   Class {cls} : {cnt:,} samples ({pct:.1f}%)")

    # Diagnosis
    print("\n[VERIFY] Diagnosis:")
    val_acc_e1   = history.history['val_accuracy'][0]
    train_acc_e1 = history.history['accuracy'][0]

    if val_acc_e1 > 0.88:
        print(f"   [!!] Val acc at epoch 1 = {val_acc_e1*100:.1f}% — suspiciously high.")
        print(f"        -> Likely spatial leakage. Confirm patch_config has spatial_split=True.")
    else:
        print(f"   [OK] Val acc at epoch 1 = {val_acc_e1*100:.1f}% — looks normal.")

    if dominant_pct > 70.0:
        print(f"   [!!] One class = {dominant_pct:.1f}% of test set — imbalance issue.")
    else:
        print(f"   [OK] Class distribution OK (dominant: {dominant_pct:.1f}%).")

    if val_acc_e1 > train_acc_e1 + 0.05:
        print(f"   [!!] Val acc > Train acc at epoch 1 — strong leakage signal.")
    else:
        print(f"   [OK] Train/Val gap at epoch 1 looks normal.")

    print("\n[VERIFY] Done. Check models/ for all saved outputs.")
    print("=" * 57)


def train_hybridsn():
    print("=========================================================")
    print("   HybridSN - VRAM-SAFE TRAINING (3.4 GB GPU)            ")
    print("=========================================================")

    # ── Step 1: Load patches ───────────────────────────────────────
    print("\n[STEP 1] Loading patches...")
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    X_path      = os.path.join(base_dir, 'data/X_patches.npy')
    Y_path      = os.path.join(base_dir, 'data/Y_patches.npy')
    X_test_path = os.path.join(base_dir, 'data/X_test_patches.npy')
    Y_test_path = os.path.join(base_dir, 'data/Y_test_patches.npy')

    if not os.path.exists(X_path):
        print(f"[!!] {X_path} not found — run patch_extractor.py first.")
        return

    X = np.load(X_path)
    y = np.load(Y_path).astype(np.float32)

    print(f"   X shape : {X.shape}   ({X.nbytes/1e9:.2f} GB in CPU RAM)")
    print(f"   Labels  : {np.unique(y)}")

    patch_size = X.shape[1]
    n_bands    = X.shape[3]

    # Patch overlap sanity check
    print(f"\n[VERIFY] Patch overlap check:")
    print(f"   Total train patches : {X.shape[0]:,}")
    print(f"   Patch size          : {patch_size}x{patch_size}")
    print(f"   Spectral bands      : {n_bands}")
    if X.shape[0] > 50000:
        print("   [!!] High patch count — confirm spatial_split=True in patch_config.json")
    else:
        print("   [OK] Patch count looks reasonable.")

    # ── Label mapping ─────────────────────────────────────────────
    unique_classes = np.unique(y)
    label_map   = {float(old): int(new) for new, old in enumerate(unique_classes)}
    reverse_map = {int(new): float(old) for new, old in enumerate(unique_classes)}
    num_classes = len(unique_classes)

    patch_config_path = os.path.join(base_dir, 'data/patch_config.json')
    patch_config = {}
    if os.path.exists(patch_config_path):
        with open(patch_config_path) as f:
            patch_config = json.load(f)
        spatial = patch_config.get('spatial_split', False)
        print(f"\n   Spatial split in patch_config : {spatial}")
        if not spatial:
            print("   [!!] WARNING: spatial_split=False in patch_config — leakage risk!")

    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'label_map.json'), 'w') as f:
        json.dump({
            'label_map'       : {str(k): v for k, v in label_map.items()},
            'reverse_map'     : {str(k): v for k, v in reverse_map.items()},
            'unique_classes'  : [float(c) for c in unique_classes],
            'num_classes'     : num_classes,
            'background_label': 0.0,
            'patch_size'      : int(patch_size),
            'n_pca_bands'     : int(n_bands),
            **patch_config
        }, f, indent=2)

    print(f"\n   Label map : {label_map}")
    print(f"   ✔ Saved -> models/label_map.json")

    y_int = np.array([label_map[float(lbl)] for lbl in y])

    # ── Class weights ─────────────────────────────────────────────
    weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_int), y=y_int
    )
    cw = dict(enumerate(weights))
    print(f"   Class weights : {cw}")

    # ── Train/Test split ──────────────────────────────────────────
    y_enc = to_categorical(y_int, num_classes)

    if os.path.exists(X_test_path):
        # Spatially pre-split by patch_extractor — zero leakage
        print("\n   [OK] Spatial pre-split test set found — no leakage.")
        X_train = X[..., np.newaxis]
        y_train = y_enc
        X_test_raw = np.load(X_test_path)
        Y_test_raw = np.load(Y_test_path).astype(np.float32)
        y_test_int = np.array([label_map.get(float(l), 0) for l in Y_test_raw])
        X_test     = X_test_raw[..., np.newaxis]
        y_test     = to_categorical(y_test_int, num_classes)
        del X_test_raw
    else:
        # Fallback random split — re-run patch_extractor to fix properly
        print("\n   [!!] No spatial test set — falling back to random split (leakage risk).")
        print("        Re-run patch_extractor.py to fix this.")
        X_all = X[..., np.newaxis]
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_enc, test_size=0.2, random_state=42, stratify=y_int
        )

    print(f"\n   Train : {X_train.shape[0]:,}  |  Test : {X_test.shape[0]:,}")
    print(f"   Train RAM : {X_train.nbytes/1e9:.2f} GB")

    # ── Streaming datasets ────────────────────────────────────────
    BATCH_SIZE = 256
    print(f"\n   Building streaming datasets (batch_size={BATCH_SIZE})...")
    train_ds = make_dataset(X_train, y_train, BATCH_SIZE, shuffle=True)
    test_ds  = make_dataset(X_test,  y_test,  BATCH_SIZE, shuffle=False)

    steps_per_epoch = len(X_train) // BATCH_SIZE
    val_steps       = len(X_test)  // BATCH_SIZE
    print(f"   Steps per epoch : {steps_per_epoch}")
    print(f"   Val steps       : {val_steps}")

    # ── Step 2: Build + compile ───────────────────────────────────
    print("\n[STEP 2] Building HybridSN...")
    model = build_hybridsn(patch_size, n_bands, num_classes)
    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    # ── Callbacks ─────────────────────────────────────────────────
    saved_models_dir = os.path.join(models_dir, 'saved_models')
    os.makedirs(saved_models_dir, exist_ok=True)
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=4, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=15,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(saved_models_dir, 'best_model.h5'),
                        monitor='val_accuracy',
                        save_best_only=True, verbose=1),
    ]

    # ── Step 3: Train ─────────────────────────────────────────────
    print("\n[STEP 3] Training...")
    print("   Batches stream CPU -> GPU one at a time.\n")

    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=100,
        validation_data=test_ds,
        validation_steps=val_steps,
        class_weight=cw,
        callbacks=callbacks
    )

    # ── Step 4: Save curves + model ───────────────────────────────
    print("\n[STEP 4] Saving training curves...")
    plot_training_curves(history, models_dir)

    model.save(os.path.join(saved_models_dir, 'final_model.h5'))
    best = max(history.history['val_accuracy'])
    print(f"\n[OK] Training complete!  Best val accuracy : {best*100:.1f}%")

    # ── Step 5: Full verification ─────────────────────────────────
    run_verification(model, test_ds, val_steps, history, models_dir)

    return history


if __name__ == "__main__":
    train_hybridsn()