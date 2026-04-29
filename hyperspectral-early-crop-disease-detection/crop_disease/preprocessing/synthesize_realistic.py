import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def generate_perlin_like_noise(shape, scale=10):
    noise = np.random.rand(*shape)
    return gaussian_filter(noise, sigma=scale)

def create_realistic_class_map(height, width):
    
    # base plant probability map
    plant_noise = generate_perlin_like_noise((height, width), scale=12)

    # threshold to create plant mask
    plant_mask = plant_noise > 0.45

    class_map = np.zeros((height, width))

    # disease center
    cy = np.random.randint(height//3, 2*height//3)
    cx = np.random.randint(width//3, 2*width//3)

    yy, xx = np.mgrid[:height, :width]
    dist = np.sqrt((yy-cy)**2 + (xx-cx)**2)

    # gradual disease spread
    disease_prob = np.exp(-dist/25)

    # stressed transition region
    stressed_prob = np.exp(-dist/45)

    # assign classes
    class_map[plant_mask] = 1

    stressed_zone = (plant_mask) & (stressed_prob > 0.35)
    class_map[stressed_zone] = 2

    disease_zone = (plant_mask) & (disease_prob > 0.55)
    class_map[disease_zone] = 3

    return class_map

def correlated_spectral_noise(num_pixels, num_bands):

    # Base noise reduced to 0.005
    base_noise = np.random.normal(0, 0.005, (num_pixels, 10))
    
    smooth_noise = gaussian_filter(base_noise, sigma=2, axes=1)
    
    expanded_noise = np.repeat(smooth_noise, num_bands//10 + 1, axis=1)
    
    return expanded_noise[:, :num_bands]

def synthesize_realistic_data(output_folder='crop_disease/data/synthetic_realistic/'):

    print("\nGenerating realistic hyperspectral eggplant image...\n")

    # Fix paths for extraction
    sigs_path = 'crop_disease/data/class_signatures.npy'
    list_path = 'crop_disease/data/class_list.npy'
    
    if not os.path.exists(sigs_path):
        print(f"❌ Error: {sigs_path} not found.")
        return

    sigs = np.load(sigs_path)
    classes = np.load(list_path)

    num_classes, num_bands, num_channels = sigs.shape

    height, width = 256, 256

    class_map = create_realistic_class_map(height, width)

    X_synth = np.zeros((height, width, num_bands, num_channels))

    # illumination gradient reduced to ±5% (0.95 to 1.05)
    gradient = np.linspace(0.95, 1.05, width)
    gradient_map = np.tile(gradient, (height,1))

    for idx, cls in enumerate(classes):

        mask = class_map == cls
        n_pixels = np.sum(mask)

        if n_pixels == 0:
            continue

        base_signature = sigs[idx]

        # spectral variability
        spectral_variation = correlated_spectral_noise(n_pixels, num_bands)
        spectral_variation = spectral_variation[..., np.newaxis]

        # lighting variability (now std=0.01)
        intensity_scale = np.random.normal(1.0, 0.01, (n_pixels,1,1))

        # spatial texture variation (now std=0.005)
        texture_variation = np.random.normal(0, 0.005, (n_pixels, num_bands, num_channels))

        generated_pixels = (
            base_signature * intensity_scale
            + spectral_variation
            + texture_variation
        )

        X_synth[mask] = generated_pixels

    # apply illumination gradient
    X_synth *= gradient_map[:,:,np.newaxis,np.newaxis]

    os.makedirs(output_folder, exist_ok=True)

    np.save(os.path.join(output_folder,'X_synthetic.npy'),
            X_synth.reshape(-1, num_bands, num_channels).astype(np.float32))

    np.save(os.path.join(output_folder,'Y_synthetic.npy'),
            class_map.flatten().astype(np.float32))

    np.save(os.path.join(output_folder,'image_dims.npy'),
            [height, width, num_bands])

    # visualize
    plt.figure(figsize=(6,5))
    plt.imshow(class_map, cmap='jet')
    plt.title("Realistic Synthetic Class Map")
    plt.colorbar()
    plt.savefig(os.path.join(output_folder,'synthetic_map.png'))
    plt.close()

    print("Saved realistic synthetic dataset to:", output_folder)


if __name__ == "__main__":
    synthesize_realistic_data()
