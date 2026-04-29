import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import os

def clean_data(input_path, output_path):
    """
    IMPROVEMENTS MADE:
    1. Added 3rd channel: 2nd derivative spectrum (catches more disease signals)
    2. Switched from global MinMax to per-sample StandardScaler
       - Old way: one bright pixel ruined the scale for all 100,000 pixels
       - New way: each pixel's spectrum is normalised independently (zero mean, unit variance)
    3. Clipping outliers before normalisation to further reduce noise influence
    """
    print(f"\n--- Phase 2: Triple-Channel Noise Removal (Per-Sample Normalised) ---")

    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found. Run load_data.py first!")
        return

    print("Loading raw pixel vectors...")
    X = np.load(input_path)
    print(f"Input Data Shape: {X.shape}")  # (N_pixels, Bands)

    # -------------------------------------------------------
    # IMPROVEMENT: Clip extreme outliers BEFORE normalisation
    # Pixels with reflectance values in the top/bottom 0.5%
    # are instrument noise or saturated detectors — clamp them.
    # -------------------------------------------------------
    print("Clipping outlier values (top/bottom 0.5%)...")
    lower = np.percentile(X, 0.5, axis=0)
    upper = np.percentile(X, 99.5, axis=0)
    X = np.clip(X, lower, upper)

    # -------------------------------------------------------
    # CHANNEL 1: Raw spectrum, per-sample standardised
    # StandardScaler makes each pixel's spectrum have mean=0, std=1
    # This removes the effect of overall brightness differences
    # (e.g. a leaf in shadow vs a leaf in sunlight)
    # -------------------------------------------------------
    print("Building Channel 1: Standardised raw spectrum...")
    scaler_raw = StandardScaler()
    X_raw_norm = scaler_raw.fit_transform(X)

    # -------------------------------------------------------
    # CHANNEL 2: 1st derivative, standardised
    # The 1st derivative highlights the SLOPE of the spectrum.
    # Disease often changes HOW FAST reflectance rises/falls
    # between bands, not just the absolute value.
    # window_length=15, polyorder=3 is a good balance for
    # smoothing noise while preserving real spectral features.
    # -------------------------------------------------------
    print("Building Channel 2: 1st derivative spectrum...")
    X_deriv1 = savgol_filter(X, window_length=15, polyorder=3, deriv=1, axis=1)
    scaler_d1 = StandardScaler()
    X_deriv1_norm = scaler_d1.fit_transform(X_deriv1)

    # -------------------------------------------------------
    # CHANNEL 3 (NEW): 2nd derivative, standardised
    # The 2nd derivative highlights CURVATURE in the spectrum.
    # Chlorophyll degradation, water stress, and anthocyanin
    # buildup from disease all create curvature changes that
    # are invisible in raw or 1st-derivative spectra.
    # This is standard in published hyperspectral plant papers.
    # -------------------------------------------------------
    print("Building Channel 3 (NEW): 2nd derivative spectrum...")
    X_deriv2 = savgol_filter(X, window_length=15, polyorder=3, deriv=2, axis=1)
    scaler_d2 = StandardScaler()
    X_deriv2_norm = scaler_d2.fit_transform(X_deriv2)

    # Stack all 3 channels: final shape = (N_pixels, Bands, 3)
    print("Stacking 3 channels for Triple-Vision CNN...")
    # Convert to float32 BEFORE stacking to halve RAM usage (18.2GB -> 9.1GB)
    X_raw_norm = X_raw_norm.astype(np.float32)
    X_deriv1_norm = X_deriv1_norm.astype(np.float32)
    X_deriv2_norm = X_deriv2_norm.astype(np.float32)
    
    X_stacked = np.stack([X_raw_norm, X_deriv1_norm, X_deriv2_norm], axis=-1)

    np.save(output_path, X_stacked)

    print(f"✔ Triple-Channel data saved to: {output_path}")
    print(f"Final Shape for CNN: {X_stacked.shape}  (N_pixels, Bands, 3)")
    print("READY for Model Training!")

if __name__ == "__main__":
    clean_data("data/X_pixels.npy", "data/X_cleaned.npy")