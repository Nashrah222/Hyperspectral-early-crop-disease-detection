"""
main.py  —  root folder

NEW PIPELINE ORDER:
  1. load_data.py        — load raw ENVI cube → X_pixels.npy, Y_labels.npy
  2. patch_extractor.py  — PCA + spatial patches → X_patches.npy, Y_patches.npy
  3. hybridsn_model.py   — train HybridSN on patches
  4. evaluate.py         — classification report + confusion matrix
  5. generate_map.py     — heatmap dashboard
  6. region_props.py     — bounding box detection

FILES TO DELETE BEFORE RUNNING (old pipeline, incompatible):
  data/X_cleaned.npy
  data/X_patches.npy          (if you ran an older version)
  data/Y_patches.npy          (if you ran an older version)
  models/saved_models/best_model.h5
  models/saved_models/final_model.h5
  models/label_map.json
"""

import os
import time

# ================================================================
# IMPORTANT: These environment variables MUST be set before
# importing TensorFlow or any modules that import TensorFlow.
# ================================================================
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES']   = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']   = '2'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

# Add CUDA and local folder to PATH for DLL discovery
_extra_paths = [
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin',
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp',
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin',
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp',
    os.path.dirname(os.path.abspath(__file__)), 
]
os.environ['PATH'] = ';'.join(_extra_paths) + ';' + os.environ.get('PATH', '')
# ================================================================

from preprocessing import load_data
from preprocessing.patch_extractor import extract_patches
from models.hybridsn_model import train_hybridsn
from results import generate_map, evaluate
from object_classification import region_props


def main():
    # Ensure we are running in the script's directory so relative paths work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"   [INFO] Working directory: {os.getcwd()}")
    print("=========================================================")
    print("   HYPERSPECTRAL CROP DISEASE DETECTION — HybridSN       ")
    print("=========================================================")
    start = time.time()

    # ── STEP 1: Load raw data ──────────────────────────────────────
    print("\n[STEP 1/6] Loading raw hyperspectral data...")
    input_hdr = "data/raw/Eggplant_Reflectance_Data.hdr"
    gt_hdr    = "data/raw/Eggplant_N2_Concentration_GT.hdr"

    if os.path.exists('data/X_pixels.npy') and os.path.exists('data/Y_labels.npy'):
        print("   ✔ Found X_pixels.npy and Y_labels.npy - skipping Step 1.")
    elif os.path.exists(input_hdr) and os.path.exists(gt_hdr):
        load_data.load_data(input_hdr, gt_hdr, "data/")
    else:
        print(f"[!!] HDR files not found in data/raw/ and .npy files missing.")
        print(f"     Path check: {os.path.abspath(input_hdr)}")
        return

    # ── STEP 2: PCA + Patch extraction ────────────────────────────
    # Skipped if patches already exist (saves ~5 minutes on reruns)
    print("\n[STEP 2/6] Extracting spatial patches with PCA...")
    if os.path.exists('data/X_patches.npy') and os.path.exists('data/Y_patches.npy'):
        print("   ✔ Patches already exist — skipping extraction.")
        print("   (Delete data/X_patches.npy to force re-extraction)")
    else:
        extract_patches(
            hdr_data_path='data/X_pixels.npy',
            labels_path='data/Y_labels.npy',
            dims_path='data/image_dims.npy',
            output_dir='data/'
        )

    # ── STEP 3: Train HybridSN ────────────────────────────────────
    print("\n[STEP 3/6] Training HybridSN model...")
    model_path = 'models/saved_models/best_model.h5'

    if os.path.exists(model_path):
        print("   ✔ Trained model found — skipping training.")
        print("   (Delete models/saved_models/best_model.h5 to retrain)")
    else:
        train_hybridsn()

    # ── STEP 4: Evaluate ──────────────────────────────────────────
    print("\n[STEP 4/6] Evaluating performance...")
    evaluate.evaluate_performance()

    # ── STEP 5: Generate heatmap ──────────────────────────────────
    print("\n[STEP 5/6] Generating disease heatmap...")
    generate_map.generate_stress_map()

    # ── STEP 6: Region detection ───────────────────────────────────
    print("\n[STEP 6/6] Detecting disease regions...")
    region_props.classify_regions()

    # ── Done ──────────────────────────────────────────────────────
    elapsed = time.time() - start
    print("\n=========================================================")
    print(f"   COMPLETE — total time: {elapsed/60:.1f} minutes")
    print("   Check results/ folder for all outputs:")
    print("     classification_report.txt  — F1 scores per class")
    print("     confusion_matrix.png       — per-class accuracy grid")
    print("     final_dashboard.png        — disease heatmap")
    print("     final_detection_clean.png  — bounding box map")
    print("=========================================================")


if __name__ == "__main__":
    main()

# PS D:\Projects\crop_disease> $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2"
# >> $env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp;" + $env:PATH
# >> .\gpu_env\Scripts\python.exe .\crop_disease\main.py