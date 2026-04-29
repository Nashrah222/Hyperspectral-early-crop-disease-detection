import sys
import os

# ================================================================
# MUST be set BEFORE importing TensorFlow — fixes CUDA_ERROR_NOT_PERMITTED
# on RTX 40-series laptops (and any GPU with strict memory allocation).
# ================================================================
os.environ['TF_GPU_ALLOCATOR']          = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES']      = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']      = '2'

# Add CUDA/cuDNN bin dirs to PATH so TensorFlow can find GPU libs
_cuda_paths = [
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin',
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp',
]
os.environ['PATH'] = ';'.join(_cuda_paths) + ';' + os.environ.get('PATH', '')
# ================================================================

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from skimage import measure, morphology
from preprocessing import load_data, noise_removal
from models import cnn_model
from scipy.ndimage import median_filter
from sklearn.metrics import accuracy_score, classification_report
import joblib


def run_fast_pipeline():
    print("=========================================================")
    print("   🍆 FAST CROP DISEASE DETECTION (OPTIMIZED) 🍆   ")
    print("=========================================================")
    start_time = time.time()

    # --- 1. LOAD & CLEAN ---
    print("\n[STEP 1] Checking Data...")
    if not os.path.exists("data/X_cleaned.npy"):
        print("   -> Data not found. Loading and cleaning now...")
        input_hdr = "data/raw/Eggplant_Reflectance_Data.hdr"
        gt_hdr = "data/raw/Eggplant_N2_Concentration_GT.hdr"
        load_data.load_data(input_hdr, gt_hdr, "data/")
        noise_removal.clean_data("data/X_pixels.npy", "data/X_cleaned.npy")
    else:
        print("   -> Clean data found. Loading...")

    X = np.load('data/X_cleaned.npy')
    y = np.load('data/Y_labels.npy')
    dims = np.load('data/image_dims.npy')
    height, width = dims[0], dims[1]

    # --- 2. LOAD MODEL ---
    print("\n[STEP 2] Loading AI Model...")
    model_path = 'models/saved_models/best_model.h5'
    if not os.path.exists(model_path):
        print("   -> Model not found. Training now...")
        cnn_model.train_model()

    model = tf.keras.models.load_model(model_path)

    # --- 3. PREDICT ONCE ---
    print("\n[STEP 3] Running Full Diagnosis (Happens only once!)...")
    print("   -> Building CPU Conveyor Belt for Prediction...")

    predict_dataset = tf.data.Dataset.from_tensor_slices(X).batch(128).prefetch(tf.data.AUTOTUNE)

    all_pred_probs = model.predict(predict_dataset)
    all_preds = np.argmax(all_pred_probs, axis=1)

    # --- SMOOTHING ---
    print("\n[INFO] Applying Median Filter to remove spectral noise...")

    pred_map_2d = all_preds.reshape(height, width)
    smoothed_map_2d = median_filter(pred_map_2d, size=15)
    all_preds_smoothed = smoothed_map_2d.flatten()

    print("   -> Calculating New Accuracy...")
    final_acc = accuracy_score(y, all_preds_smoothed)

    print("\n" + "=" * 50)
    print(f"🚀 FINAL SMOOTHED ACCURACY: {final_acc * 100:.2f}%")
    print("=" * 50 + "\n")

    unique_classes = np.unique(y)
    report_labels = ['Background', 'Healthy', 'Stressed', 'Diseased'] if len(unique_classes) == 4 else ['Healthy', 'Stressed', 'Diseased']

    print("--- DETAILED CLASSIFICATION REPORT ---")
    print(classification_report(y, all_preds_smoothed, target_names=report_labels))

    # --- 4. DASHBOARD ---
    print("\n[STEP 4] Generating Dashboard...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- CONFUSION MATRIX ---
    print("   -> Rendering Confusion Matrix...")
    _, y_true_sample, _, y_pred_sample = train_test_split(
        y, all_preds_smoothed, test_size=0.1, random_state=42
    )

    cm = confusion_matrix(y_true_sample, y_pred_sample)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=report_labels, yticklabels=report_labels)

    axes[0].set_title(f"Smoothed Accuracy ({final_acc * 100:.1f}%)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # --- STRESS MAP ---
    print("   -> Rendering Stress Map...")
    axes[1].imshow(smoothed_map_2d, cmap='viridis')
    axes[1].set_title("Smoothed Pixel-wise Stress Map")
    axes[1].axis('off')

    # --- OBJECT DETECTION ---
    print("   -> Detecting Disease Regions...")

    disease_class = np.max(all_preds_smoothed)
    binary_mask = (smoothed_map_2d == disease_class).astype(int)

    cleaned_mask = morphology.opening(binary_mask, morphology.square(5))
    cleaned_mask = morphology.closing(cleaned_mask, morphology.square(45))

    labels_img = measure.label(cleaned_mask, connectivity=2)
    props = measure.regionprops(labels_img)

    large_regions = [p for p in props if p.area > 2000]
    large_regions.sort(key=lambda x: x.area, reverse=True)

    axes[2].imshow(smoothed_map_2d, cmap='viridis')
    axes[2].set_title(f"Final Detection: {len(large_regions)} Major Areas")
    axes[2].axis('off')

    for i, prop in enumerate(large_regions):
        minr, minc, maxr, maxc = prop.bbox

        rect = mpatches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor='red',
            linewidth=2
        )
        axes[2].add_patch(rect)

        if i < 3:
            axes[2].text(
                minc, minr - 10,
                f"Area {i + 1}",
                color='white',
                fontweight='bold',
                bbox=dict(facecolor='red', alpha=0.5)
            )

    # --- SAVE ---
    plt.tight_layout()
    save_path = 'results/final_dashboard.png'
    plt.savefig(save_path)

    print(f"\n✅ Dashboard saved to {save_path}")
    print("   (Close the window to finish the program)")

    end_time = time.time()
    print(f"   ⏱ Total Execution Time: {end_time - start_time:.2f} seconds")

    plt.show()


if __name__ == "__main__":
    run_fast_pipeline()