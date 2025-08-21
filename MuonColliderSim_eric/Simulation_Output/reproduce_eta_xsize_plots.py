import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Directory containing the parquet files
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Load all signal and background label and recon2D files
def load_all_data(data_dir, prefix):
    """Load and concatenate all files with the given prefix"""
    label_files = sorted([f for f in os.listdir(data_dir) if f.startswith(f"labels{prefix}") and f.endswith(".parquet")])
    recon2d_files = sorted([f for f in os.listdir(data_dir) if f.startswith(f"recon2D{prefix}") and f.endswith(".parquet")])
    
    if not label_files or not recon2d_files:
        return pd.DataFrame(), pd.DataFrame()
    
    # Load and concatenate label files
    truth_data = pd.concat([pd.read_parquet(os.path.join(data_dir, f)) for f in label_files], ignore_index=True)
    
    # Load and concatenate recon2D files
    recon2d_data = pd.concat([pd.read_parquet(os.path.join(data_dir, f)) for f in recon2d_files], ignore_index=True)
    
    return truth_data, recon2d_data

print("Loading signal data...")
truthsig, recon2Dsig = load_all_data(DATA_DIR, "sig")
print("Loading background data...")
truthbib, recon2Dbib = load_all_data(DATA_DIR, "bib")

print(f"Signal - Truth: {truthsig.shape}, Recon2D: {recon2Dsig.shape}")
print(f"Background - Truth: {truthbib.shape}, Recon2D: {recon2Dbib.shape}")

# Ensure both dataframes have the same number of rows
min_rows_sig = min(len(truthsig), len(recon2Dsig))
min_rows_bib = min(len(truthbib), len(recon2Dbib))

# Truncate to matching lengths
truthsig = truthsig.iloc[:min_rows_sig].copy()
recon2Dsig = recon2Dsig.iloc[:min_rows_sig].copy()
truthbib = truthbib.iloc[:min_rows_bib].copy()
recon2Dbib = recon2Dbib.iloc[:min_rows_bib].copy()

print(f"After truncation - Signal: {len(truthsig)}, Background: {len(truthbib)}")

# Reshape clusters for 2D pixel data (n_samples, 13, 21) - following plot_signal_data.py
clustersSig = recon2Dsig.to_numpy().reshape(recon2Dsig.shape[0], 13, 21)
clustersBib = recon2Dbib.to_numpy().reshape(recon2Dbib.shape[0], 13, 21)

# Calculate eta from z-global for both signal and background
if 'z-global' in truthsig.columns:
    theta_sig = np.arctan(30 / truthsig['z-global'])
    truthsig['eta'] = -np.log(np.tan(theta_sig / 2))

if 'z-global' in truthbib.columns:
    theta_bib = np.arctan(30 / truthbib['z-global'])
    truthbib['eta'] = -np.log(np.tan(theta_bib / 2))

# Calculate x-size and y-size following plot_signal_data.py methodology
# x-size: number of nonzero pixels per row, then max over rows
xSizesSig = np.count_nonzero(clustersSig, axis=2).max(axis=1)
xSizesBib = np.count_nonzero(clustersBib, axis=2).max(axis=1)

# Create pastel red colormap
# Using a custom colormap that goes from white to pastel red
colors = ['#ffffff', '#ffe6e6', '#ffcccc', '#ffb3b3', '#ff9999', '#ff8080', '#ff6666', '#ff4d4d', '#ff3333', '#ff1a1a', '#ff0000']
n_bins = 100  # Discretizes the interpolation into bins
pastel_red_cmap = mcolors.LinearSegmentedColormap.from_list('pastel_red', colors, N=n_bins)

# Create the plot following MuCDataSetPlots.ipynb structure but only top row
fig, ax = plt.subplots(1, 2, sharex='col', sharey='row', gridspec_kw={'hspace': 0.05, 'wspace': 0.03}, figsize=(10, 4))

# Filter out invalid values
valid_sig = np.isfinite(truthsig['eta']) & np.isfinite(xSizesSig) & (xSizesSig > 0)
valid_bib = np.isfinite(truthbib['eta']) & np.isfinite(xSizesBib) & (xSizesBib > 0)

# BIB plot (left)
if valid_bib.sum() > 0:
    ax[0].hist2d(truthbib['eta'][valid_bib], xSizesBib[valid_bib], 
                bins=[30, np.arange(0, 9, 1)], cmap=pastel_red_cmap)
    ax[0].set_ylabel('x-size (# pixels)', fontsize=15)
    ax[0].set_title("BIB", fontsize=15)
    ax[0].set_xlabel('η', fontsize=15)

# Signal plot (right)
if valid_sig.sum() > 0:
    ax[1].hist2d(truthsig['eta'][valid_sig], xSizesSig[valid_sig], 
                bins=[30, np.arange(0, 9, 1)], cmap=pastel_red_cmap)
    ax[1].set_title("Signal", fontsize=15)
    ax[1].set_xlabel('η', fontsize=15)

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "plots", "eta_xsize_bib_signal_pastel_red.png"), dpi=300, bbox_inches='tight')
plt.show()

print(f"Plot saved to {os.path.join(DATA_DIR, 'plots', 'eta_xsize_bib_signal_pastel_red.png')}")