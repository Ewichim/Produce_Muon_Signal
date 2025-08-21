# PixelSim2 - Muon Collider Signal Track Production

This repository contains simulation tools for producing and analyzing Muon Collider signal tracks using pixel detector simulations. The workflow consists of multiple stages from track generation to detailed pixel response simulation.

## Repository Structure

- **`produceSmartPixMuC/`** - Initial track generation and preprocessing
- **`MuonColliderSim/`** - Main simulation pipeline and analysis
- **`pixelav/`** - Pixel detector response simulation (PixelAV)

## Quick Start: Producing Muon Collider Signal Tracks

### Prerequisites

Ensure you have access to the Muon Collider software stack:
```bash
source /cvmfs/muoncollider.cern.ch/release/2.8-patch2/setup.sh
```

### Step 1: Generate Track Lists

Navigate to the track generation directory and set up the environment:

```bash
cd produceSmartPixMuC/
source setup.sh
```

**For Signal Tracks:**
```bash
# Generate muon gun sample in LCIO format
source particle_gun.sh

# Run GEANT4 detector simulation
source detector_sim.sh

# Convert to PixelAV input format
python study_signal.py
```

**For Background (BIB) Tracks:**
```bash
python study_bib.py
```

### Step 2: Run Pixel Response Simulation

Navigate to the simulation directory:

```bash
cd ../MuonColliderSim/
```

Run the pixel response simulation using PixelAV:

```bash
python launchMuC.py [options]
```

**Key Options:**
- `-o, --outDir`: Output directory (default: `./Simulation_Output`)
- `-j, --ncpu`: Number of CPU cores to use (default: 20)
- `-p, --pixelAVdir`: Path to PixelAV directory (default: `../pixelav`)

### Step 3: Analysis and Visualization

The simulation outputs are automatically converted to Parquet format for analysis. Use the provided Jupyter notebooks for visualization:

- `MuCDataSetPlots.ipynb` - Dataset overview and statistics
- `plotting_clusters.ipynb` - Cluster analysis
- `hit_time.ipynb` - Timing analysis

## Output Structure

The simulation produces:
- **Track lists** - Text files with particle trajectories for PixelAV input
- **Parquet files** - Processed detector response data for analysis
- **ROOT files** - Raw simulation output (if needed)

## Key Scripts and Files

### produceSmartPixMuC/
- `study_signal.py` - Main signal track processing
- `study_bib.py` - Background track processing
- `particle_gun.sh` - Muon gun simulation setup
- `detector_sim.sh` - GEANT4 simulation execution

### MuonColliderSim/
- `launchMuC.py` - Main simulation launcher with parallel processing
- `write_parquet.py` - Output format conversion
- `processing/` - Data processing utilities

### pixelav/
- Pixel detector response simulation engine
- Contains C code for detailed pixel modeling

## Parallel Processing

The simulation supports parallel processing for efficiency:
- Use `particle_gun_parallel.sh` and `detector_sim_parallel.sh` for multi-core track generation
- `launchMuC.py` includes built-in parallelization for PixelAV simulations