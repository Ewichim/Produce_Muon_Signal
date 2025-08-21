import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Directory paths
SIGNAL_DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIB_DATA_DIR = "/home/youeric/PixelSim2/MuonColliderSim_eric/Simulation_Output"
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Load all signal data
def load_all_signal_data(data_dir):
    label_files = sorted([f for f in os.listdir(data_dir) if f.startswith("labelssig") and f.endswith(".parquet")])
    recon2d_files = sorted([f for f in os.listdir(data_dir) if f.startswith("recon2Dsig") and f.endswith(".parquet")])
    
    truthsig = pd.concat([pd.read_parquet(os.path.join(data_dir, f)) for f in label_files], ignore_index=True)
    recon2Dsig = pd.concat([pd.read_parquet(os.path.join(data_dir, f)) for f in recon2d_files], ignore_index=True)
    return truthsig, recon2Dsig

# Load all BIB data
def load_all_bib_data(data_dir):
    label_files = sorted([f for f in os.listdir(data_dir) if f.startswith("labelsbib") and f.endswith(".parquet")])
    recon2d_files = sorted([f for f in os.listdir(data_dir) if f.startswith("recon2Dbib") and f.endswith(".parquet")])
    
    truthbib = pd.concat([pd.read_parquet(os.path.join(data_dir, f)) for f in label_files], ignore_index=True)
    recon2Dbib = pd.concat([pd.read_parquet(os.path.join(data_dir, f)) for f in recon2d_files], ignore_index=True)
    return truthbib, recon2Dbib

print("Loading Signal data...")
truthsig, recon2Dsig = load_all_signal_data(SIGNAL_DATA_DIR)

print("Loading BIB data...")
truthbib, recon2Dbib = load_all_bib_data(BIB_DATA_DIR)

# Debug: Check shapes
print(f"Signal - truthsig shape: {truthsig.shape}, recon2Dsig shape: {recon2Dsig.shape}")
print(f"BIB - truthbib shape: {truthbib.shape}, recon2Dbib shape: {recon2Dbib.shape}")

# Ensure each dataset has matching rows
min_rows_sig = min(len(truthsig), len(recon2Dsig))
min_rows_bib = min(len(truthbib), len(recon2Dbib))

print(f"Using {min_rows_sig} signal rows and {min_rows_bib} BIB rows")

# Truncate both to matching lengths
truthsig = truthsig.iloc[:min_rows_sig].copy()
recon2Dsig = recon2Dsig.iloc[:min_rows_sig].copy()
truthbib = truthbib.iloc[:min_rows_bib].copy()
recon2Dbib = recon2Dbib.iloc[:min_rows_bib].copy()

# Reshape clusters for 2D pixel data (n_samples, 13, 21)
clustersSig = recon2Dsig.to_numpy().reshape(recon2Dsig.shape[0], 13, 21)
clustersBib = recon2Dbib.to_numpy().reshape(recon2Dbib.shape[0], 13, 21)

# Create a custom pastel red colormap
def create_pastel_red_cmap():
    # Define colors from white to a pastel/soft red
    colors_list = ['#ffffff', '#ffe6e6', '#ffcccc', '#ffb3b3', '#ff9999', '#ff8080', '#ff6666', '#ff4d4d', '#ff3333', '#e62e2e']
    n_bins = 256
    cmap = colors.LinearSegmentedColormap.from_list('pastel_red', colors_list, N=n_bins)
    return cmap

pastel_red_cmap = create_pastel_red_cmap()

# Calculate x-sizes and y-sizes for both datasets
xSizesSig = np.count_nonzero(clustersSig, axis=2).max(axis=1)  # Max non-zero pixels in x-direction
ySizesSig = np.count_nonzero(clustersSig, axis=1).max(axis=1)  # Max non-zero pixels in y-direction
xSizesBib = np.count_nonzero(clustersBib, axis=2).max(axis=1)
ySizesBib = np.count_nonzero(clustersBib, axis=1).max(axis=1)

# --- Plot 1: Side-by-side comparison of z-global vs x-size ---
fig, ax = plt.subplots(1, 2, figsize=(18, 8))

# Check if z-global column exists in both datasets
if 'z-global' in truthbib.columns and 'z-global' in truthsig.columns:
    z_global_bib = truthbib['z-global']
    z_global_sig = truthsig['z-global']
    
    # Left panel: BIB
    mask_bib = ~np.isnan(z_global_bib) & ~np.isnan(xSizesBib)
    mask_bib = mask_bib & np.isfinite(z_global_bib) & np.isfinite(xSizesBib)
    
    ax[0].hist2d(z_global_bib[mask_bib], xSizesBib[mask_bib], bins=[30, np.arange(0,9,1)], cmap=pastel_red_cmap)
    ax[0].set_title("BIB - X Size", fontsize=63)
    ax[0].set_xlabel('z-global [mm]', fontsize=50)
    ax[0].set_ylabel('x-size (# pixels)', fontsize=50)
    ax[0].tick_params(axis='both', which='major', labelsize=40)
    
    # Right panel: Signal
    mask_sig = ~np.isnan(z_global_sig) & ~np.isnan(xSizesSig)
    mask_sig = mask_sig & np.isfinite(z_global_sig) & np.isfinite(xSizesSig)
    
    ax[1].hist2d(z_global_sig[mask_sig], xSizesSig[mask_sig], bins=[30, np.arange(0,9,1)], cmap=pastel_red_cmap)
    ax[1].set_title("Signal - X Size", fontsize=63)
    ax[1].set_xlabel('z-global [mm]', fontsize=50)
    ax[1].set_ylabel('x-size (# pixels)', fontsize=50)
    ax[1].tick_params(axis='both', which='major', labelsize=40)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "bib_signal_zglobal_vs_xsize_comparison_with_ysize.png"), dpi=300, bbox_inches='tight')
    plt.close()

# --- Plot 2: Side-by-side comparison of z-global vs y-size ---
fig, ax = plt.subplots(1, 2, figsize=(18, 8))

if 'z-global' in truthbib.columns and 'z-global' in truthsig.columns:
    # Left panel: BIB
    mask_bib_y = ~np.isnan(z_global_bib) & ~np.isnan(ySizesBib)
    mask_bib_y = mask_bib_y & np.isfinite(z_global_bib) & np.isfinite(ySizesBib)
    
    ax[0].hist2d(z_global_bib[mask_bib_y], ySizesBib[mask_bib_y], bins=[30, np.arange(0,9,1)], cmap=pastel_red_cmap)
    ax[0].set_title("BIB - Y Size", fontsize=63)
    ax[0].set_xlabel('z-global [mm]', fontsize=50)
    ax[0].set_ylabel('y-size (# pixels)', fontsize=50)
    ax[0].tick_params(axis='both', which='major', labelsize=40)
    
    # Right panel: Signal
    mask_sig_y = ~np.isnan(z_global_sig) & ~np.isnan(ySizesSig)
    mask_sig_y = mask_sig_y & np.isfinite(z_global_sig) & np.isfinite(ySizesSig)
    
    ax[1].hist2d(z_global_sig[mask_sig_y], ySizesSig[mask_sig_y], bins=[30, np.arange(0,9,1)], cmap=pastel_red_cmap)
    ax[1].set_title("Signal - Y Size", fontsize=63)
    ax[1].set_xlabel('z-global [mm]', fontsize=50)
    ax[1].set_ylabel('y-size (# pixels)', fontsize=50)
    ax[1].tick_params(axis='both', which='major', labelsize=40)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "bib_signal_zglobal_vs_ysize_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

# --- Plot 3: 2x2 grid showing both x-size and y-size comparisons ---
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

if 'z-global' in truthbib.columns and 'z-global' in truthsig.columns:
    # Top row: X-size comparisons
    # Top-left: BIB X-size
    axes[0,0].hist2d(z_global_bib[mask_bib], xSizesBib[mask_bib], bins=[30, np.arange(0,9,1)], cmap=pastel_red_cmap)
    axes[0,0].set_title("BIB - X Size", fontsize=38)
    axes[0,0].set_xlabel('z-global [mm]', fontsize=31)
    axes[0,0].set_ylabel('x-size (# pixels)', fontsize=31)
    axes[0,0].tick_params(axis='both', which='major', labelsize=23)
    
    # Top-right: Signal X-size
    axes[0,1].hist2d(z_global_sig[mask_sig], xSizesSig[mask_sig], bins=[30, np.arange(0,9,1)], cmap=pastel_red_cmap)
    axes[0,1].set_title("Signal - X Size", fontsize=38)
    axes[0,1].set_xlabel('z-global [mm]', fontsize=31)
    axes[0,1].set_ylabel('x-size (# pixels)', fontsize=31)
    axes[0,1].tick_params(axis='both', which='major', labelsize=23)
    
    # Bottom row: Y-size comparisons
    # Bottom-left: BIB Y-size
    axes[1,0].hist2d(z_global_bib[mask_bib_y], ySizesBib[mask_bib_y], bins=[30, np.arange(0,9,1)], cmap=pastel_red_cmap)
    axes[1,0].set_title("BIB - Y Size", fontsize=38)
    axes[1,0].set_xlabel('z-global [mm]', fontsize=31)
    axes[1,0].set_ylabel('y-size (# pixels)', fontsize=31)
    axes[1,0].tick_params(axis='both', which='major', labelsize=23)
    
    # Bottom-right: Signal Y-size
    axes[1,1].hist2d(z_global_sig[mask_sig_y], ySizesSig[mask_sig_y], bins=[30, np.arange(0,9,1)], cmap=pastel_red_cmap)
    axes[1,1].set_title("Signal - Y Size", fontsize=38)
    axes[1,1].set_xlabel('z-global [mm]', fontsize=31)
    axes[1,1].set_ylabel('y-size (# pixels)', fontsize=31)
    axes[1,1].tick_params(axis='both', which='major', labelsize=23)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "bib_signal_zglobal_vs_xysize_grid_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved:")
    print(f"1. X-size only: {os.path.join(PLOT_DIR, 'bib_signal_zglobal_vs_xsize_comparison_with_ysize.png')}")
    print(f"2. Y-size only: {os.path.join(PLOT_DIR, 'bib_signal_zglobal_vs_ysize_comparison.png')}")
    print(f"3. 2x2 grid: {os.path.join(PLOT_DIR, 'bib_signal_zglobal_vs_xysize_grid_comparison.png')}")
    
    print(f"\nBIB statistics:")
    print(f"  z-global range: {z_global_bib.min():.2f} to {z_global_bib.max():.2f} mm")
    print(f"  x-size range: {xSizesBib.min()} to {xSizesBib.max()} pixels")
    print(f"  y-size range: {ySizesBib.min()} to {ySizesBib.max()} pixels")
    
    print(f"\nSignal statistics:")
    print(f"  z-global range: {z_global_sig.min():.2f} to {z_global_sig.max():.2f} mm")
    print(f"  x-size range: {xSizesSig.min()} to {xSizesSig.max()} pixels")
    print(f"  y-size range: {ySizesSig.min()} to {ySizesSig.max()} pixels")
    
else:
    print("Error: 'z-global' column not found in one or both datasets")
    if 'z-global' not in truthbib.columns:
        print("BIB columns:", list(truthbib.columns))
    if 'z-global' not in truthsig.columns:
        print("Signal columns:", list(truthsig.columns))