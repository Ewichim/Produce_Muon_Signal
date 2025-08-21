import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.ticker as ticker

# Directory containing the parquet files
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

def load_all_data(data_dir, file_prefix):
    """Load and concatenate all files with the given prefix"""
    files = sorted([f for f in os.listdir(data_dir) if f.startswith(file_prefix) and f.endswith(".parquet")])
    if not files:
        return pd.DataFrame()
    data_frames = []
    for f in files:
        try:
            df = pd.read_parquet(os.path.join(data_dir, f))
            data_frames.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

print("Loading data...")
# Load all signal and background data
truthsig = load_all_data(DATA_DIR, "labelssig")
truthbib = load_all_data(DATA_DIR, "labelsbib") 
recon3Dsig = load_all_data(DATA_DIR, "recon3Dsig")
recon3Dbib = load_all_data(DATA_DIR, "recon3Dbib")

print(f"Signal truth shape: {truthsig.shape}")
print(f"Background truth shape: {truthbib.shape}")
print(f"Signal recon3D shape: {recon3Dsig.shape}")
print(f"Background recon3D shape: {recon3Dbib.shape}")

# Ensure matching lengths
min_sig = min(len(truthsig), len(recon3Dsig))
min_bib = min(len(truthbib), len(recon3Dbib))

truthsig = truthsig.iloc[:min_sig].copy()
recon3Dsig = recon3Dsig.iloc[:min_sig].copy()
truthbib = truthbib.iloc[:min_bib].copy()
recon3Dbib = recon3Dbib.iloc[:min_bib].copy()

print(f"After truncation - Signal: {len(truthsig)}, Background: {len(truthbib)}")

# Reshape 3D clusters (assuming 13x21x20 based on 5460 features)
clusters3Dsig = recon3Dsig.to_numpy().reshape(recon3Dsig.shape[0], 13, 21, 20)
clusters3Dbib = recon3Dbib.to_numpy().reshape(recon3Dbib.shape[0], 13, 21, 20)

# Calculate eta from z-global for signal data
if 'z-global' in truthsig.columns:
    theta = np.arctan(30 / truthsig['z-global'])
    truthsig['eta'] = -np.log(np.tan(theta / 2))

# Define pt categories
def categorize_pt(pt_values):
    """Categorize pt values into Low, Mid, High"""
    low_mask = pt_values < 33.33
    high_mask = pt_values > 66.67
    mid_mask = ~(low_mask | high_mask)
    return low_mask, mid_mask, high_mask

# Create pt categories for both signal and background
sig_low, sig_mid, sig_high = categorize_pt(truthsig['pt'])
bib_low, bib_mid, bib_high = categorize_pt(truthbib['pt'])

print(f"Signal categories - Low: {sig_low.sum()}, Mid: {sig_mid.sum()}, High: {sig_high.sum()}")
print(f"Background categories - Low: {bib_low.sum()}, Mid: {bib_mid.sum()}, High: {bib_high.sum()}")

# Calculate charge fraction vs y-pixels
def calculate_charge_fraction_vs_y(clusters, y_pixels=21):
    """Calculate fraction of total cluster charge vs y-pixel position"""
    # Sum over x and z dimensions to get charge vs y
    charge_vs_y = np.sum(clusters, axis=(1, 3))  # Sum over x(dim 1) and z(dim 3)
    
    # Calculate total charge for each cluster
    total_charge = np.sum(charge_vs_y, axis=1, keepdims=True)
    
    # Avoid division by zero
    total_charge[total_charge == 0] = 1
    
    # Calculate fraction
    fraction_vs_y = charge_vs_y / total_charge
    
    return fraction_vs_y

# Calculate charge fractions
sig_fractions = calculate_charge_fraction_vs_y(clusters3Dsig)
bib_fractions = calculate_charge_fraction_vs_y(clusters3Dbib)

# Calculate eta for background data too
if 'z-global' in truthbib.columns:
    theta_bib = np.arctan(30 / truthbib['z-global'])
    truthbib['eta'] = -np.log(np.tan(theta_bib / 2))

# Create the combined plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Plot 1: Fraction of total cluster charge vs y-pixels (without y-local differentiation)
y_pixels = np.arange(1, 22)  # 1 to 21 pixels

# Calculate mean fractions for each pt category with proper formatting like MuonFilter.ipynb
categories = ['Low', 'Mid', 'High']
sig_masks = [sig_low, sig_mid, sig_high]
bib_masks = [bib_low, bib_mid, bib_high]
colors_bib = ['red', 'red', 'red']
colors_sig = ['blue', 'blue', 'blue']

for i, (cat, sig_mask, bib_mask) in enumerate(zip(categories, sig_masks, bib_masks)):
    if sig_mask.sum() > 0:
        sig_mean = np.mean(sig_fractions[sig_mask], axis=0)
        sig_std = np.std(sig_fractions[sig_mask], axis=0)
        # Use step plot like in MuonFilter.ipynb
        ax1.step(y_pixels, sig_mean, where="mid", label=f'sig {cat}', color=colors_sig[i], linewidth=2)
        ax1.errorbar(y_pixels, sig_mean, yerr=sig_std, fmt='o', color=colors_sig[i], alpha=0.5, markersize=3)
    
    if bib_mask.sum() > 0:
        bib_mean = np.mean(bib_fractions[bib_mask], axis=0)
        bib_std = np.std(bib_fractions[bib_mask], axis=0)
        # Use step plot like in MuonFilter.ipynb
        ax1.step(y_pixels, bib_mean, where="mid", label=f'bib {cat}', color=colors_bib[i], linewidth=2)
        ax1.errorbar(y_pixels, bib_mean, yerr=bib_std, fmt='o', color=colors_bib[i], alpha=0.5, markersize=3)

ax1.set_xlabel('y [pixels]', fontsize=12)
ax1.set_ylabel('Fraction of total cluster charge', fontsize=12)
ax1.set_title('Combined Analysis (No y-local differentiation)', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1, 21)

# Plot 2: x-size vs eta comparison (signal vs background) with difference heatmap
if len(truthsig) > 0 and 'eta' in truthsig.columns and len(truthbib) > 0 and 'eta' in truthbib.columns:
    # Convert 3D clusters to 2D-like for x-size calculation (following original plot_signal_data.py)
    # Sum over z-dimension to get 2D representation, then calculate x-size like original
    clusters2d_sig = np.sum(clusters3Dsig, axis=3)  # Sum over z to get (n_samples, 13, 21)
    clusters2d_bib = np.sum(clusters3Dbib, axis=3)
    
    # Calculate x-size exactly like in plot_signal_data.py: count nonzero per row, then max over rows
    x_sizes_sig = np.count_nonzero(clusters2d_sig, axis=2).max(axis=1)
    x_sizes_bib = np.count_nonzero(clusters2d_bib, axis=2).max(axis=1)
    
    # Filter valid data
    valid_sig = np.isfinite(truthsig['eta']) & np.isfinite(x_sizes_sig) & (x_sizes_sig > 0)
    valid_bib = np.isfinite(truthbib['eta']) & np.isfinite(x_sizes_bib) & (x_sizes_bib > 0)
    
    if valid_sig.sum() > 0 and valid_bib.sum() > 0:
        # Create histograms for both signal and background
        bins_eta = 30
        bins_xsize = np.arange(0, 9, 1)  # Following original plot_signal_data.py
        
        h_sig, xedges, yedges = np.histogram2d(truthsig['eta'][valid_sig], x_sizes_sig[valid_sig], 
                                              bins=[bins_eta, bins_xsize])
        h_bib, _, _ = np.histogram2d(truthbib['eta'][valid_bib], x_sizes_bib[valid_bib], 
                                    bins=[bins_eta, bins_xsize])
        
        # Normalize to get densities for fair comparison
        h_sig_norm = h_sig / h_sig.sum() if h_sig.sum() > 0 else h_sig
        h_bib_norm = h_bib / h_bib.sum() if h_bib.sum() > 0 else h_bib
        
        # Calculate difference (signal - background)
        h_diff = h_sig_norm - h_bib_norm
        
        # Plot the difference heatmap
        X, Y = np.meshgrid(xedges, yedges)
        im = ax2.pcolormesh(X, Y, h_diff.T, cmap='RdBu_r', shading='auto', 
                           vmin=-np.abs(h_diff).max(), vmax=np.abs(h_diff).max())
        
        ax2.set_xlabel('η', fontsize=12)
        ax2.set_ylabel('x-size (# pixels)', fontsize=12)
        ax2.set_title('Signal - Background Difference\n(x-size vs η)', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2, label='Normalized Density Difference')
        cbar.ax.tick_params(labelsize=10)
    else:
        ax2.text(0.5, 0.5, 'No valid data for x-size vs η plot', 
                ha='center', va='center', transform=ax2.transAxes)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "combined_analysis_plot.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"Combined analysis plot saved to {os.path.join(PLOT_DIR, 'combined_analysis_plot.png')}")

# Create separate eta vs x-size plots for signal and background
if len(truthsig) > 0 and 'eta' in truthsig.columns and len(truthbib) > 0 and 'eta' in truthbib.columns:
    fig_eta, ((ax_sig, ax_bib), (ax_diff, ax_empty)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Signal plot
    if valid_sig.sum() > 0:
        im_sig = ax_sig.hist2d(truthsig['eta'][valid_sig], x_sizes_sig[valid_sig], 
                              bins=[30, np.arange(0, 9, 1)], cmap='Blues')
        ax_sig.set_title('Signal: x-size vs η', fontsize=14)
        ax_sig.set_ylabel('x-size (# pixels)', fontsize=12)
        plt.colorbar(im_sig[3], ax=ax_sig, label='Count')
    
    # Background plot
    if valid_bib.sum() > 0:
        im_bib = ax_bib.hist2d(truthbib['eta'][valid_bib], x_sizes_bib[valid_bib], 
                              bins=[30, np.arange(0, 9, 1)], cmap='Reds')
        ax_bib.set_title('Background: x-size vs η', fontsize=14)
        ax_bib.set_ylabel('x-size (# pixels)', fontsize=12)
        plt.colorbar(im_bib[3], ax=ax_bib, label='Count')
    
    # Difference plot (repeated from main plot for clarity)
    if valid_sig.sum() > 0 and valid_bib.sum() > 0:
        X, Y = np.meshgrid(xedges, yedges)
        im_diff = ax_diff.pcolormesh(X, Y, h_diff.T, cmap='RdBu_r', shading='auto',
                                    vmin=-np.abs(h_diff).max(), vmax=np.abs(h_diff).max())
        ax_diff.set_title('Signal - Background Difference', fontsize=14)
        ax_diff.set_xlabel('η', fontsize=12)
        ax_diff.set_ylabel('x-size (# pixels)', fontsize=12)
        plt.colorbar(im_diff, ax=ax_diff, label='Normalized Density Difference')
    
    # Remove empty subplot
    ax_empty.remove()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "eta_vs_xsize_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Eta vs x-size comparison plot saved to {os.path.join(PLOT_DIR, 'eta_vs_xsize_comparison.png')}")

# Also create individual plots for comparison
fig, ax = plt.subplots(figsize=(12, 6))

# Recreate just the charge fraction plot with improved formatting
for i, (cat, sig_mask, bib_mask) in enumerate(zip(categories, sig_masks, bib_masks)):
    if sig_mask.sum() > 0:
        sig_mean = np.mean(sig_fractions[sig_mask], axis=0)
        sig_std = np.std(sig_fractions[sig_mask], axis=0)
        # Use step plot like in MuonFilter.ipynb
        ax.step(y_pixels, sig_mean, where="mid", label=f'sig {cat}', color=colors_sig[i], linewidth=2)
        ax.errorbar(y_pixels, sig_mean, yerr=sig_std, fmt='o', color=colors_sig[i], alpha=0.5, markersize=3)
    
    if bib_mask.sum() > 0:
        bib_mean = np.mean(bib_fractions[bib_mask], axis=0)
        bib_std = np.std(bib_fractions[bib_mask], axis=0)
        # Use step plot like in MuonFilter.ipynb
        ax.step(y_pixels, bib_mean, where="mid", label=f'bib {cat}', color=colors_bib[i], linewidth=2)
        ax.errorbar(y_pixels, bib_mean, yerr=bib_std, fmt='o', color=colors_bib[i], alpha=0.5, markersize=3)

ax.set_xlabel('y [pixels]', fontsize=12)
ax.set_ylabel('Fraction of total cluster charge', fontsize=12)
ax.set_title('Charge Fraction Analysis (No y-local differentiation)', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 21)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "charge_fraction_no_ylocal.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"Charge fraction plot saved to {os.path.join(PLOT_DIR, 'charge_fraction_no_ylocal.png')}")
print("Analysis complete!")