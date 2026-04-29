import numpy as np
import os
import matplotlib.pyplot as plt

def synthesize_data(output_folder='data/synthetic/'):
    print("=========================================================")
    print("   EGGPLANT HYPERSPECTRAL DATA SYNTHESIZER               ")
    print("=========================================================")
    
    # 1. Load Mean Signatures
    if not os.path.exists('crop_disease/data/class_signatures.npy'):
        print("❌ Error: crop_disease/data/class_signatures.npy not found.")
        return
        
    sigs = np.load('crop_disease/data/class_signatures.npy') # (4, 277, 3)
    classes = np.load('crop_disease/data/class_list.npy')     # [0, 1, 2, 3]
    num_bands = sigs.shape[1]
    
    # 2. Setup Image Dimensions
    height, width = 200, 200
    print(f"Generating {height}x{width} synthetic image...")
    
    # 3. Create a Class Map (0=BG, 1=Healthy, 2=Stressed, 3=Diseased)
    # We create some "blobs" of plant tissue
    y_synth = np.zeros((height, width), dtype=np.float32)
    
    # Create a large plant blob (Healthy)
    yy, xx = np.mgrid[:height, :width]
    dist = np.sqrt((yy - 100)**2 + (xx - 100)**2)
    y_synth[dist < 80] = 1.0
    
    # Add a "Stressed" area
    y_synth[(dist < 40) & (xx > 100)] = 2.0
    
    # Add a "Diseased" spot
    y_synth[(dist < 20) & (yy > 110)] = 3.0
    
    # 4. Generate Spectral Data
    X_synth = np.zeros((height, width, num_bands, 3), dtype=np.float32)
    
    for c_idx, cls_val in enumerate(classes):
        mask = (y_synth == cls_val)
        n_pixels = np.sum(mask)
        if n_pixels == 0: continue
        
        # Base signature
        base_sig = sigs[c_idx]
        
        # Add random variation for each pixel
        # 1. Variation in absolute intensity (lighting)
        intensity_scale = np.random.normal(1.0, 0.05, (n_pixels, 1, 1))
        
        # 2. Additive Gaussian noise (sensor noise)
        noise = np.random.normal(0, 0.01, (n_pixels, num_bands, 3))
        
        # Apply to pixels
        X_synth[mask] = (base_sig * intensity_scale) + noise
        
    # 5. Save Results
    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, 'X_synthetic.npy'), X_synth.reshape(-1, num_bands, 3))
    np.save(os.path.join(output_folder, 'Y_synthetic.npy'), y_synth.flatten())
    np.save(os.path.join(output_folder, 'image_dims.npy'), [height, width, num_bands])
    
    print(f"✔ Synthetic data saved to: {output_folder}")
    
    # Optional: Visualise the class map
    plt.imshow(y_synth)
    plt.title("Synthetic Class Map")
    plt.colorbar(label='Class')
    plt.savefig(os.path.join(output_folder, 'synthetic_map.png'))
    print(f"✔ Visualization saved to: {output_folder}synthetic_map.png")

if __name__ == "__main__":
    synthesize_data()
