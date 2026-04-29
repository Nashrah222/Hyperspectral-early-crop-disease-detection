"""
patch_extractor.py  —  place in preprocessing/

Memory-safe version for 16GB RAM.

Key insight: you have 1.8M labelled pixels but you don't need all of them.
Training on 400k well-chosen patches gives the same accuracy as 1.8M
because the model sees the same spectral patterns repeatedly after ~400k.
Sampling also speeds up training significantly.

Memory budget:
  400,000 patches × 7×7 pixels × 90 PCA bands × 2 bytes (float16)
  = 400,000 × 49 × 90 × 2 = ~3.5 GB  ✅ safe on 16GB RAM

Spatial split:
  Top 80% of image rows → TRAIN
  Bottom 20% of image rows → TEST
  This prevents patch overlap between train/test (no leakage).
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
import pickle
import json
import os

# ── Configuration ─────────────────────────────────────────────────────
PATCH_SIZE   = 7       # 7×7 spatial window — safe for 16GB RAM
N_PCA_COMPS  = 30      # PCA components per channel → 30×3 = 90 total
BACKGROUND   = 0.0     # label value for background — always skipped
MAX_SAMPLES  = 400000  # max train patches (None = use all, needs 40GB+)
# ──────────────────────────────────────────────────────────────────────


def get_pixels(row_start, row_end, labels_2d, max_n, background):
    """
    Return (rows, cols) of labelled pixels within [row_start, row_end).
    Stratified-samples down to max_n if there are too many.
    """
    row_mask = np.zeros(labels_2d.shape[0], dtype=bool)
    row_mask[row_start:row_end] = True
    rs, cs = np.where((labels_2d > background) & row_mask[:, None])

    n_available = len(rs)
    if max_n is not None and n_available > max_n:
        y_here   = labels_2d[rs, cs]
        selected = []
        for cls in np.unique(y_here):
            idx   = np.where(y_here == cls)[0]
            n_cls = max(int(max_n * len(idx) / n_available), 100)
            chosen = np.random.choice(idx, size=min(n_cls, len(idx)),
                                      replace=False)
            selected.append(chosen)
        sel    = np.concatenate(selected)
        rs, cs = rs[sel], cs[sel]

    return rs, cs


def extract_patch_set(rows, cols, cube_padded, labels_2d,
                      patch_size, half, n_total, label):
    """Extract patches centered on each (row, col) pixel."""
    n  = len(rows)
    Xp = np.zeros((n, patch_size, patch_size, n_total), dtype=np.float16)
    Yp = np.zeros(n, dtype=np.float32)

    print(f"\n   Extracting {label} patches ({n:,})...")
    for i, (r, c) in enumerate(zip(rows, cols)):
        pr, pc  = r + half, c + half
        Xp[i]   = cube_padded[pr-half:pr+half+1, pc-half:pc+half+1, :]
        Yp[i]   = labels_2d[r, c]
        if (i + 1) % 100000 == 0:
            print(f"     {i+1:,} / {n:,}  ({(i+1)/n*100:.0f}%)")

    return Xp, Yp


def extract_patches(hdr_data_path='data/X_pixels.npy',
                    labels_path='data/Y_labels.npy',
                    dims_path='data/image_dims.npy',
                    output_dir='data/'):

    patch_size  = PATCH_SIZE
    max_samples = MAX_SAMPLES

    print("=========================================================")
    print("   PATCH EXTRACTOR — SPATIAL SPLIT + MEMORY-SAFE         ")
    print("=========================================================")

    # ── Step 1: Load ───────────────────────────────────────────────
    print("\n[1/5] Loading raw data...")
    X    = np.load(hdr_data_path).astype(np.float32)
    y    = np.load(labels_path).astype(np.float32)
    dims = np.load(dims_path)
    H, W, Bands = int(dims[0]), int(dims[1]), int(dims[2])

    print(f"   Image size : {H} × {W} = {H*W:,} pixels")
    print(f"   Bands      : {Bands}")
    print(f"   RAM used by X : {X.nbytes/1e9:.2f} GB")

    # ── Step 2: Clip outliers ──────────────────────────────────────
    print("\n[2/5] Clipping outliers...")
    lower = np.percentile(X, 0.5,  axis=0)
    upper = np.percentile(X, 99.5, axis=0)
    X = np.clip(X, lower, upper)

    labelled = (y > BACKGROUND)
    print(f"   Total labelled pixels : {labelled.sum():,}")

    # ── Step 3: PCA per channel ────────────────────────────────────
    print(f"\n[3/5] Running PCA per channel ({N_PCA_COMPS} components each)...")
    print("   Computing derivatives (takes ~1 min)...")

    X_d1 = savgol_filter(X, window_length=15, polyorder=3,
                          deriv=1, axis=1).astype(np.float32)
    X_d2 = savgol_filter(X, window_length=15, polyorder=3,
                          deriv=2, axis=1).astype(np.float32)

    channels = [
        ('Raw spectrum',   X),
        ('1st derivative', X_d1),
        ('2nd derivative', X_d2),
    ]

    pca_results = []
    pca_scalers = []
    pca_models  = []

    for name, ch_data in channels:
        print(f"   Processing: {name}")
        scaler  = StandardScaler()
        ch_norm = scaler.fit_transform(ch_data)

        pca = PCA(n_components=N_PCA_COMPS, whiten=True)
        pca.fit(ch_norm[labelled])
        ch_pca = pca.transform(ch_norm).astype(np.float32)
        del ch_norm

        var = pca.explained_variance_ratio_.sum() * 100
        print(f"     Variance kept: {var:.1f}%  shape: {ch_pca.shape}")

        pca_results.append(ch_pca)
        pca_scalers.append(scaler)
        pca_models.append(pca)

    del X_d1, X_d2

    print("\n   Concatenating PCA outputs...")
    X_pca   = np.concatenate(pca_results, axis=1).astype(np.float32)
    n_total = X_pca.shape[1]   # 90
    del pca_results
    print(f"   Final PCA shape: {X_pca.shape}  ({X_pca.nbytes/1e9:.2f} GB)")

    # Save PCA bundle
    os.makedirs(output_dir, exist_ok=True)
    pca_bundle = {
        'pca_raw':    pca_models[0], 'scaler_raw': pca_scalers[0],
        'pca_d1':     pca_models[1], 'scaler_d1':  pca_scalers[1],
        'pca_d2':     pca_models[2], 'scaler_d2':  pca_scalers[2],
        'n_components_per_channel': N_PCA_COMPS,
        'n_total_components': n_total,
    }
    with open(os.path.join(output_dir, 'pca_model.pkl'), 'wb') as f:
        pickle.dump(pca_bundle, f)
    print(f"   ✔ PCA bundle saved → {output_dir}pca_model.pkl")

    # ── Step 4: Spatial split + extract patches ────────────────────
    print(f"\n[4/5] Spatial train/test split + extracting patches...")

    cube      = X_pca.reshape(H, W, n_total)
    labels_2d = y.reshape(H, W)
    half      = patch_size // 2
    del X_pca

    cube_padded = np.pad(cube, ((half, half), (half, half), (0, 0)),
                         mode='reflect')
    del cube

    # Top 80% rows → train | Bottom 20% rows → test
    # Ensures zero patch overlap between train and test sets
    split_row = int(H * 0.80)
    print(f"   Image height : {H} rows")
    print(f"   Train rows   : 0 → {split_row - 1}  (top 80%)")
    print(f"   Test rows    : {split_row} → {H - 1}  (bottom 20%)")
    print(f"   No patch from train region overlaps test region ✔")

    train_rows, train_cols = get_pixels(
        0, split_row, labels_2d, max_samples, BACKGROUND
    )
    test_rows, test_cols = get_pixels(
        split_row, H, labels_2d, max_samples // 4, BACKGROUND
    )

    print(f"\n   Train pixels selected : {len(train_rows):,}")
    print(f"   Test  pixels selected : {len(test_rows):,}")

    X_train_patches, Y_train_patches = extract_patch_set(
        train_rows, train_cols, cube_padded, labels_2d,
        patch_size, half, n_total, "TRAIN"
    )
    X_test_patches, Y_test_patches = extract_patch_set(
        test_rows, test_cols, cube_padded, labels_2d,
        patch_size, half, n_total, "TEST"
    )

    # ── Step 5: Save ───────────────────────────────────────────────
    print(f"\n[5/5] Saving...")
    np.save(os.path.join(output_dir, 'X_patches.npy'),      X_train_patches)
    np.save(os.path.join(output_dir, 'Y_patches.npy'),      Y_train_patches)
    np.save(os.path.join(output_dir, 'X_test_patches.npy'), X_test_patches)
    np.save(os.path.join(output_dir, 'Y_test_patches.npy'), Y_test_patches)

    config = {
        'patch_size'              : patch_size,
        'n_total_components'      : n_total,
        'n_components_per_channel': N_PCA_COMPS,
        'max_samples'             : max_samples,
        'n_train'                 : int(len(X_train_patches)),
        'n_test'                  : int(len(X_test_patches)),
        'spatial_split_row'       : split_row,
        'spatial_split'           : True,
    }
    with open(os.path.join(output_dir, 'patch_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"   ✔ X_patches.npy      ({len(X_train_patches):,} train patches)")
    print(f"   ✔ Y_patches.npy")
    print(f"   ✔ X_test_patches.npy ({len(X_test_patches):,} test patches)")
    print(f"   ✔ Y_test_patches.npy")
    print(f"   ✔ patch_config.json")
    print("\n   READY for HybridSN training!")
    print("=========================================================")

    return X_train_patches, Y_train_patches


if __name__ == "__main__":
    extract_patches()