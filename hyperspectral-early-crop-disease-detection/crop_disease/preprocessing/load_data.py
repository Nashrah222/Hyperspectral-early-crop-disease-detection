#load_data.py

# from spectral import open_image
# import numpy as np

# def load_hyperspectral_image(hdr_path):
#     img = open_image(hdr_path)
#     cube = img.load()
#     return np.array(cube)
import spectral.io.envi as envi
import numpy as np
import os

def load_data(hdr_path, gt_path, save_folder):
    """
    Responsibilities:
    1. Read .hdr hyperspectral files (Image and Ground Truth)
    2. Convert Hyperspectral cube -> pixel vectors (Flattening)
    3. Store data as X_pixels.npy and Y_labels.npy
    """
    print(f"--- Person 1: Loading Data ---")
    
    # --- PART 1: Load the Image (X) ---
    print(f"Reading Image: {hdr_path}")
    try:
        # We use envi.open because your files are in ENVI format (.hdr)
        img = envi.open(hdr_path)
        raw_cube = img.load() 
        print(f"✔ Original Cube Shape (H, W, Bands): {raw_cube.shape}")
    except Exception as e:
        print(f"❌ Error loading Image: {e}")
        return

    # --- PART 2: Load the Ground Truth (Y) ---
    print(f"Reading Ground Truth: {gt_path}")
    try:
        gt_img = envi.open(gt_path)
        gt_cube = gt_img.load()
        # GT is usually 3D (H, W, 1), we squeeze it to 2D (H, W)
        gt_data = gt_cube.squeeze() 
        print(f"✔ Ground Truth Shape: {gt_data.shape}")
    except Exception as e:
        print(f"❌ Error loading GT: {e}")
        return

    # --- PART 3: Flatten (The "One-Pixel-One-Vector" Logic) ---
    height, width, bands = raw_cube.shape
    
    # Flatten X: (Height * Width, Bands)
    X_flat = raw_cube.reshape((height * width, bands))
    
    # Flatten Y: (Height * Width)
    Y_flat = gt_data.flatten()
    
    print(f"✔ Flattened X Shape: {X_flat.shape}")
    print(f"✔ Flattened Y Shape: {Y_flat.shape}")

    # --- PART 4: Save to Disk ---
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    np.save(os.path.join(save_folder, 'X_pixels.npy'), X_flat)
    np.save(os.path.join(save_folder, 'Y_labels.npy'), Y_flat)
    
    # We save dimensions to reconstruct the image later (Step 4 of your project)
    np.save(os.path.join(save_folder, 'image_dims.npy'), np.array([height, width, bands]))
    
    print(f"✔ Data saved successfully to: {save_folder}")

# --- EXECUTION BLOCK (Runs when you press Play) ---
if __name__ == "__main__":
    # These match the files you showed in your screenshot
    input_hdr = "data/raw/Eggplant_Reflectance_Data.hdr"
    gt_hdr = "data/raw/Eggplant_N2_Concentration_GT.hdr"
    
    # Save processed files to 'data/' folder
    output_folder = "data/"
    
    load_data(input_hdr, gt_hdr, output_folder)