import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Directory containing the parquet files
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Load all signal label and recon2D files
def load_all_signal_data(data_dir):
    label_files = sorted([f for f in os.listdir(data_dir) if f.startswith("labelssig") and f.endswith(".parquet")])
    recon2d_files = sorted([f for f in os.listdir(data_dir) if f.startswith("recon2Dsig") and f.endswith(".parquet")])
    
    truthsig = pd.concat([pd.read_parquet(os.path.join(data_dir, f)) for f in label_files], ignore_index=True)
    recon2Dsig = pd.concat([pd.read_parquet(os.path.join(data_dir, f)) for f in recon2d_files], ignore_index=True)
    return truthsig, recon2Dsig

truthsig, recon2Dsig = load_all_signal_data(DATA_DIR)

# Debug: Check shapes to identify the mismatch
print(f"truthsig shape: {truthsig.shape}")
print(f"recon2Dsig shape: {recon2Dsig.shape}")
print(f"Number of truthsig rows: {len(truthsig)}")
print(f"Number of recon2Dsig rows: {len(recon2Dsig)}")

# Ensure both dataframes have the same number of rows
min_rows = min(len(truthsig), len(recon2Dsig))
print(f"Using minimum number of rows: {min_rows}")

# Truncate both to the same length
truthsig = truthsig.iloc[:min_rows].copy()
recon2Dsig = recon2Dsig.iloc[:min_rows].copy()

print(f"After truncation - truthsig shape: {truthsig.shape}")
print(f"After truncation - recon2Dsig shape: {recon2Dsig.shape}")

# Reshape clusters for 2D pixel data (n_samples, 13, 21)
clustersSig = recon2Dsig.to_numpy().reshape(recon2Dsig.shape[0], 13, 21)

# Create a custom pastel red colormap
def create_pastel_red_cmap():
    # Define colors from white to a pastel/soft red
    colors_list = ['#ffffff', '#ffe6e6', '#ffcccc', '#ffb3b3', '#ff9999', '#ff8080', '#ff6666', '#ff4d4d', '#ff3333', '#e62e2e']
    n_bins = 256
    cmap = colors.LinearSegmentedColormap.from_list('pastel_red', colors_list, N=n_bins)
    return cmap

pastel_red_cmap = create_pastel_red_cmap()

# --- Plot: 2D histograms of z-global vs x-size/y-size ---
# x-size and y-size: number of nonzero pixels per row/column
xSizesSig = np.count_nonzero(clustersSig, axis=2).max(axis=1)
ySizesSig = np.count_nonzero(clustersSig, axis=1).max(axis=1)

fig, ax = plt.subplots(2, 1, figsize=(7, 7))

# Check if z-global column exists
if 'z-global' in truthsig.columns:
    z_global = truthsig['z-global']
    
    # Filter out NaN and inf values for z-global vs x-size
    mask_x = ~np.isnan(z_global) & ~np.isnan(xSizesSig)
    mask_x = mask_x & np.isfinite(z_global) & np.isfinite(xSizesSig)
    
    # Top panel: z-global vs x-size
    ax[0].hist2d(z_global[mask_x], xSizesSig[mask_x], bins=[30, np.arange(0,9,1)], cmap=pastel_red_cmap)
    ax[0].set_title("Signal", fontsize=15)
    ax[0].set_ylabel('x-size (# pixels)', fontsize=15)
    
    # Filter out NaN and inf values for z-global vs y-size
    mask_y = ~np.isnan(z_global) & ~np.isnan(ySizesSig)
    mask_y = mask_y & np.isfinite(z_global) & np.isfinite(ySizesSig)
    
    # Bottom panel: z-global vs y-size
    ax[1].hist2d(z_global[mask_y], ySizesSig[mask_y], bins=[30, np.arange(0,14,1)], cmap=pastel_red_cmap)
    ax[1].set_xlabel('z-global [mm]', fontsize=15)
    ax[1].set_ylabel('y-size (# pixels)', fontsize=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "signal_zglobal_vs_size_2d.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved as: {os.path.join(PLOT_DIR, 'signal_zglobal_vs_size_2d.png')}")
    print(f"z-global range: {z_global.min():.2f} to {z_global.max():.2f} mm")
    print(f"x-size range: {xSizesSig.min()} to {xSizesSig.max()} pixels")
    print(f"y-size range: {ySizesSig.min()} to {ySizesSig.max()} pixels")
    
else:
    print("Error: 'z-global' column not found in the data")
    print("Available columns:", list(truthsig.columns))