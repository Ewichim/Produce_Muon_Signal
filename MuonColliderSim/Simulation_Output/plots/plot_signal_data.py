import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from particle import PDGID

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

# --- Derived quantities ---
# eta from z-global
if 'z-global' in truthsig.columns:
    theta = np.arctan(30 / truthsig['z-global'])
    truthsig['eta'] = -np.log(np.tan(theta / 2))

# cotAlpha, cotBeta may already be present, but recalculate if not
if 'cotAlpha' not in truthsig.columns and {'n_x', 'n_z'}.issubset(truthsig.columns):
    truthsig['cotAlpha'] = truthsig['n_x'] / truthsig['n_z']
if 'cotBeta' not in truthsig.columns and {'n_y', 'n_z'}.issubset(truthsig.columns):
    truthsig['cotBeta'] = truthsig['n_y'] / truthsig['n_z']

# --- Plot 1: cotAlpha, cotBeta, number_eh_pairs, nPixels ---
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,8))

ax[0,0].hist(truthsig['cotAlpha'], bins=40, histtype='step', color='g', align='mid', density=True, label="signal")
ax[0,0].set_xlabel('cot(α)')
ax[0,0].set_ylabel('Track Density')
ax[0,0].set_xlim(-7.5, 7.5)
ax[0,0].legend()

ax[1,0].hist(truthsig['cotBeta'], bins=40, histtype='step', color='g', align='mid', density=True, label="signal")
ax[1,0].set_xlabel('cot(β)')
ax[1,0].set_ylabel('Track Density')
ax[1,0].set_xlim(-8, 8)
ax[1,0].legend()

ax[0,1].hist(truthsig['number_eh_pairs'], bins=40, histtype='step', color='g', align='mid', density=True, label="signal")
ax[0,1].set_xlabel('Number of eh pairs')
ax[0,1].set_ylabel('Track Density')
ax[0,1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0e}'.format(x)))
ax[0,1].set_xlim(0, 120000)
ax[0,1].legend()

# nPixels: number of nonzero pixels in each cluster
nPixelssig = np.count_nonzero(clustersSig, axis=(1,2))
ax[1,1].hist(nPixelssig, bins=30, histtype='step', color='g', align='mid', density=True, label="signal")
ax[1,1].set_xlabel('Number of pixels')
ax[1,1].set_ylabel('Track Density')
ax[1,1].legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "signal_summary_histograms.png"))
plt.close()

# --- Plot 2: 2D histograms of eta vs x-size/y-size ---
# x-size and y-size: number of nonzero pixels per row/column
xSizesSig = np.count_nonzero(clustersSig, axis=2).max(axis=1)
ySizesSig = np.count_nonzero(clustersSig, axis=1).max(axis=1)

fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'hspace': 0.05, 'wspace': 0.03}, figsize=(10,7))

# Before each hist2d call, filter out NaN and inf values for both axes
mask_96 = ~np.isnan(truthsig['eta']) & ~np.isnan(xSizesSig)
mask_96 = mask_96 & np.isfinite(truthsig['eta']) & np.isfinite(xSizesSig)
ax[0,1].hist2d(truthsig['eta'][mask_96], xSizesSig[mask_96], bins=[30, np.arange(0,9,1)], cmap='Blues')
ax[0,1].set_title("Signal", fontsize=15)

mask_98 = ~np.isnan(truthsig['eta']) & ~np.isnan(ySizesSig)
mask_98 = mask_98 & np.isfinite(truthsig['eta']) & np.isfinite(ySizesSig)
ax[1,1].hist2d(truthsig['eta'][mask_98], ySizesSig[mask_98], bins=[30, np.arange(0,14,1)], cmap='Blues')
ax[1,1].set_xlabel('η', fontsize=15)

ax[0,1].set_ylabel('x-size (# pixels)', fontsize=15)
ax[1,1].set_ylabel('y-size (# pixels)', fontsize=15)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "signal_eta_vs_size_2d.png"))
plt.close()

# --- Plot 3: 2D histograms of y-local vs x-size/y-size ---
fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'hspace': 0.05, 'wspace': 0.03}, figsize=(10,7))

# Before each hist2d call, filter out NaN and inf values for both axes
mask_111 = ~np.isnan(truthsig['y-local']) & ~np.isnan(xSizesSig)
mask_111 = mask_111 & np.isfinite(truthsig['y-local']) & np.isfinite(xSizesSig)
ax[0,1].hist2d(truthsig['y-local'][mask_111], xSizesSig[mask_111], bins=[30, np.arange(0,9,1)], cmap='Blues')
ax[0,1].set_title("Signal", fontsize=15)

mask_113 = ~np.isnan(truthsig['y-local']) & ~np.isnan(ySizesSig)
mask_113 = mask_113 & np.isfinite(truthsig['y-local']) & np.isfinite(ySizesSig)
ax[1,1].hist2d(truthsig['y-local'][mask_113], ySizesSig[mask_113], bins=[30, np.arange(0,14,1)], cmap='Blues')
ax[1,1].set_xlabel('y-local [μm]', fontsize=15)

ax[0,1].set_ylabel('x-size (# pixels)', fontsize=15)
ax[1,1].set_ylabel('y-size (# pixels)', fontsize=15)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "signal_ylocal_vs_size_2d.png"))
plt.close()

# --- Plot 4: 2D histogram of number_eh_pairs vs pt ---
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
# Before each hist2d call, filter out NaN and inf values for both axes
mask_125 = ~np.isnan(truthsig['number_eh_pairs']) & ~np.isnan(truthsig['pt'])
mask_125 = mask_125 & np.isfinite(truthsig['number_eh_pairs']) & np.isfinite(truthsig['pt'])
ax.hist2d(truthsig['number_eh_pairs'][mask_125], truthsig['pt'][mask_125], bins=30, cmap='Blues')
ax.set_title("Signal", fontsize=15)
ax.set_ylabel('pt (GeV)', fontsize=15)
ax.set_xlabel('number of electron hole pairs', fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "signal_ehpairs_vs_pt_2d.png"))
plt.close()

# --- Plot 5: charge separation for low and high pt ---
# Add charge column using PDGID
truthsig['q'] = truthsig['PID'].apply(lambda pid: PDGID(int(pid)).charge if pd.notnull(pid) else np.nan)

# Low pt (<5 GeV) and high pt (>95 GeV)
truthLow = truthsig[truthsig['pt'] < 5].copy()
truthHigh = truthsig[truthsig['pt'] > 95].copy()

truthLowPos = truthLow[truthLow['q'] > 0]
truthLowNeg = truthLow[truthLow['q'] < 0]
truthHighPos = truthHigh[truthHigh['q'] > 0]
truthHighNeg = truthHigh[truthHigh['q'] < 0]

# Example: plot pt distributions for low/high pt, positive/negative charge
fig, ax = plt.subplots(1,2,figsize=(12,5))
ax[0].hist(truthLowPos['pt'], bins=30, alpha=0.7, label='Low pt, q>0')
ax[0].hist(truthLowNeg['pt'], bins=30, alpha=0.7, label='Low pt, q<0')
ax[0].set_title('Low pt (<5 GeV)')
ax[0].set_xlabel('pt (GeV)')
ax[0].set_ylabel('Tracks')
ax[0].legend()

ax[1].hist(truthHighPos['pt'], bins=30, alpha=0.7, label='High pt, q>0')
ax[1].hist(truthHighNeg['pt'], bins=30, alpha=0.7, label='High pt, q<0')
ax[1].set_title('High pt (>95 GeV)')
ax[1].set_xlabel('pt (GeV)')
ax[1].set_ylabel('Tracks')
ax[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "signal_pt_charge_separation.png"))
plt.close()

print(f"Plots saved in {PLOT_DIR}") 