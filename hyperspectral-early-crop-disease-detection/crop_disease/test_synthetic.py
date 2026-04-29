import os

# ================================================================
# MUST be set BEFORE importing TensorFlow
# ================================================================
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES']   = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']   = '2'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

# Add CUDA and local folder to PATH for DLL discovery
_extra_paths = [
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin',
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp',
    os.path.dirname(os.path.abspath(__file__)), # Location of zlibwapi.dll or other helpers
]
os.environ['PATH'] = ';'.join(_extra_paths) + ';' + os.environ.get('PATH', '')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from skimage import measure, morphology
import json
import pickle
from scipy.signal import savgol_filter

def run_synthetic_test(data_folder='data/synthetic_realistic/'):
    print("=========================================================")
    print("   REALISTIC SYNTHETIC DATA EVALUATION — ISOLATED TEST   ")
    print("=========================================================")
    
    # Base directory relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    out_dir = os.path.join(base_dir, 'results/synthetic_realistic/')
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load Resources
    print("\n[1/5] Loading Model and PCA bundle...")
    model_path = os.path.join(base_dir, 'models/saved_models/best_model.h5')
    model = tf.keras.models.load_model(model_path, compile=False)
    
    pca_path = os.path.join(base_dir, 'data/pca_model.pkl')
    with open(pca_path, 'rb') as f:
        pca_bundle = pickle.load(f)
        
    label_path = os.path.join(base_dir, 'models/label_map.json')
    with open(label_path) as f:
        lm = json.load(f)
    
    # 2. Load Synthetic Data
    print(f"[2/5] Loading Data from {data_folder}...")
    X_path = os.path.join(base_dir, data_folder, 'X_synthetic.npy')
    Y_path = os.path.join(base_dir, data_folder, 'Y_synthetic.npy')
    dim_path = os.path.join(base_dir, data_folder, 'image_dims.npy')
    
    X_synth = np.load(X_path) # (N, 277, 3)
    Y_synth = np.load(Y_path) # (N,)
    dims = np.load(dim_path)   # [height, width, bands]
    H, W = int(dims[0]), int(dims[1])
    
    # Reshape and extract raw channel
    X_raw = X_synth.reshape(H, W, 277, 3)[:, :, :, 0].reshape(-1, 277)
    
    # 3. PCA Transformation
    print("[3/5] Applying PCA transformation...")
    def _transform(X_chunk, pca_bundle):
        X = X_chunk.astype(np.float32)
        X_raw = pca_bundle['scaler_raw'].transform(X)
        X_raw = pca_bundle['pca_raw'].transform(X_raw).astype(np.float32)
        
        X_d1 = savgol_filter(X, window_length=15, polyorder=3, deriv=1, axis=1).astype(np.float32)
        X_d1 = pca_bundle['scaler_d1'].transform(X_d1)
        X_d1 = pca_bundle['pca_d1'].transform(X_d1).astype(np.float32)
        
        X_d2 = savgol_filter(X, window_length=15, polyorder=3, deriv=2, axis=1).astype(np.float32)
        X_d2 = pca_bundle['scaler_d2'].transform(X_d2)
        X_d2 = pca_bundle['pca_d2'].transform(X_d2).astype(np.float32)
        
        return np.concatenate([X_raw, X_d1, X_d2], axis=1).astype(np.float32)

    X_pca = _transform(X_raw, pca_bundle)
    cube = X_pca.reshape(H, W, -1)
    
    # 4. Patch Extraction & Prediction
    print("[4/5] Extracting patches and predicting...")
    half = 7 // 2
    cube_padded = np.pad(cube, ((half, half), (half, half), (0, 0)), mode='reflect')
    
    patches = []
    for r in range(H):
        pr = r + half
        for c in range(W):
            pc = c + half
            patches.append(cube_padded[pr-half:pr+half+1, pc-half:pc+half+1, :])
            
    patches = np.array(patches, dtype=np.float32)[..., np.newaxis]
    
    preds = model.predict(patches, batch_size=256, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    conf_map = np.max(preds, axis=1).reshape(H, W)
    class_map = y_pred.reshape(H, W)
    
    # 5. Reports & Visuals
    print("[5/5] Generating Isolated Reports...")
    
    label_map = {float(k): int(v) for k, v in lm['label_map'].items()}
    
    # Filter only plant pixels for the report
    mask = Y_synth > 0
    y_true_eval = np.array([label_map[float(lbl)] for lbl in Y_synth[mask]])
    y_pred_eval = y_pred[mask]
    
    accuracy = np.mean(y_true_eval == y_pred_eval)
    
    target_names = ['Healthy', 'Stressed', 'Diseased']
    
    print("\n" + "="*40)
    print(f"   OVERALL ACCURACY: {accuracy:.1%}")
    print("="*40 + "\n")
    
    report = classification_report(y_true_eval, y_pred_eval, target_names=target_names, labels=[0,1,2])
    with open(os.path.join(out_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"OVERALL ACCURACY: {accuracy:.1%}\n\n")
        f.write(report)
    
    # Visuals matching generate_map logic
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(Y_synth.reshape(H, W), cmap='jet')
    axes[0].set_title("Realistic Ground Truth")
    
    axes[1].imshow(class_map, cmap='RdYlGn_r')
    axes[1].set_title(f"Model Prediction (Acc: {accuracy:.1%})")
    
    axes[2].imshow(conf_map, cmap='viridis')
    axes[2].set_title("Confidence Map")
    
    plt.savefig(os.path.join(out_dir, 'final_dashboard.png'))
    plt.close()
    
    # Detection
    disease_mask = ((class_map >= 1) & (conf_map >= 0.65)).astype(np.uint8)
    cleaned = morphology.closing(morphology.opening(disease_mask, morphology.square(3)), morphology.square(10))
    labels_det = measure.label(cleaned)
    props = measure.regionprops(labels_det)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(class_map, cmap='RdYlGn_r')
    for p in props:
        if p.area > 200:
            r0, c0, r1, c1 = p.bbox
            ax.add_patch(mpatches.Rectangle((c0, r0), c1-c0, r1-r0, fill=False, edgecolor='red', linewidth=2))
            
    plt.savefig(os.path.join(out_dir, 'final_detection_clean.png'))
    plt.close()
    
    print(f"✅ ALL RESULTS SAVED IN: {out_dir}")

if __name__ == "__main__":
    run_synthetic_test()
