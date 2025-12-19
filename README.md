# fNIRS Anxiety Project - Shared Environment

This folder contains a device-agnostic Python environment setup that can be used across different systems (with or without CUDA acceleration).

## Table of Contents

- [Section 1: Onboarding](#section-1-onboarding)
- [Section 2: Python Environment Setup](#section-2-python-environment-setup)
  - [Prerequisites](#prerequisites)
  - [Setup Virtual Environment](#setup-virtual-environment)
  - [Installation](#installation)
    - [Option 1: CPU-Only Installation (Default)](#option-1-cpu-only-installation-default)
    - [Option 2: CUDA-Enabled Installation](#option-2-cuda-enabled-installation)
  - [Verification](#verification)
  - [Notes](#notes)
- [Section 3: Notebook Guide](#section-3-notebook-guide)
  - [1_fNIRS_data_processing.ipynb](#1_fnirs_data_processingipynb)
    - [FNIRSDataProcessorMNE](#fnirsdataprocessormne)
    - [FNIRSDataProcessorHOMER3](#fnirsdataprocessorhomer3)
    - [Example: Preprocessed fNIRS Data Folder Structure](#example-preprocessed-fnirs-data-folder-structure)
  - [2_fNIRS_graph_algorithm.ipynb](#2_fnirs_graph_algorithmipynb)
    - [1. Data Preparation](#1-data-preparation)
    - [2. Graph Model Initialization](#2-graph-model-initialization)
    - [3. 5-Fold Cross-Validation Training Pipeline](#3-5-fold-cross-validation-training-pipeline)

---

## Section 1: Onboarding

To get started with the project, complete the following steps:

1. **Set up the Python environment**: Install [uv](https://docs.astral.sh/uv/getting-started/installation/) as your Python package manager instead of the traditional `python -m venv .venv`. The `uv` package manager is significantly faster for installing Python dependencies. Refer to Section 2 for detailed Python environment setup instructions.

2. **Download the raw data**: Download the raw dataset first. Details about the dataset structure and download instructions are provided in the `README.md` file located in the 📁 `raw` folder.

3. **Explore the notebooks**: The 📁 `notebook` folder contains two key notebooks:
   - `1_fNIRS_data_processing.ipynb`: Contains the data processing pipeline for preparing data for algorithm development.
   - `2_fNIRS_graph_algorithm.ipynb`: Contains the pipeline for developing graph-based algorithms.

---

## Section 2: Python Environment Setup

### Prerequisites

- Python 3.10 or higher
- `uv` package manager (recommended) or `pip`

### Setup Virtual Environment

**IMPORTANT**: Always create a virtual environment to avoid conflicts with system packages.

#### Using uv (Recommended)

```bash
# Create a new virtual environment
uv venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### Installation

**Make sure your virtual environment is activated before proceeding!**

#### Option 1: CPU-Only Installation (Default)

This option is suitable for systems without NVIDIA GPUs or for development on laptops/workstations without CUDA.

**Step 1: Install Core Dependencies**

```bash
# Using uv (recommended)
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

**Step 2: Install PyTorch Geometric Extensions (CPU)**

These extensions are required for advanced PyTorch Geometric features and temporal GNNs:

```bash
# Using uv (recommended)
uv pip install -r requirements-pyg-extensions.txt -f https://data.pyg.org/whl/torch-2.6.0+cpu.html

# Or using pip
pip install -r requirements-pyg-extensions.txt -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
```

#### Option 2: CUDA-Enabled Installation

If you have an NVIDIA GPU and want to leverage CUDA acceleration, follow these steps:

**Step 1: Check Your CUDA Version**

```bash
# Check if NVIDIA drivers are installed
nvidia-smi

# Look for "CUDA Version" in the output (e.g., CUDA Version: 12.4)
```

**Step 2: Install PyTorch with CUDA Support**

Choose the appropriate command based on your CUDA version:

**For CUDA 12.4:**
```bash
uv pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

**For CUDA 12.1:**
```bash
uv pip install torch==2.6.0+cu121 torchvision==0.21.0+cu121 torchaudio==2.6.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8:**
```bash
uv pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

**Step 3: Install Remaining Core Dependencies**

```bash
# Install all other dependencies from requirements.txt (torch packages are already installed)
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

**Step 4: Install PyTorch Geometric Extensions with CUDA Support**

Choose the appropriate command based on your CUDA version:

**For CUDA 12.4:**
```bash
uv pip install -r requirements-pyg-extensions.txt -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

**For CUDA 12.1:**
```bash
uv pip install -r requirements-pyg-extensions.txt -f https://data.pyg.org/whl/torch-2.6.0+cu121.html
```

**For CUDA 11.8:**
```bash
uv pip install -r requirements-pyg-extensions.txt -f https://data.pyg.org/whl/torch-2.6.0+cu118.html
```

### Verification

To verify your installation, run the following Python code:

```python
import torch
import torch_geometric

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
print(f"PyTorch Geometric version: {torch_geometric.__version__}")
```

### Notes

- The default `requirements.txt` installs CPU-only versions to ensure compatibility across all systems.
- For CUDA installations, you must manually specify the appropriate wheel index for your CUDA version.
- Always check your CUDA version before installing CUDA-enabled packages to avoid compatibility issues.

---

## Section 3: Notebook Guide

### `1_fNIRS_data_processing.ipynb`

This notebook is divided into three sections:
1. **RAW fNIRS data processing using MNE-Python library**
2. **Integration of HOMER3 MATLAB preprocessed fNIRS data into MNE-Python tools** for further data preparation for algorithm development
3. **Signal similarity comparison** using various signal analysis methods such as Pearson Correlation to compare fNIRS concentration data produced by MNE-Python with data produced by HOMER3 MATLAB

Currently, we have not yet evaluated graph algorithm model performance when using fNIRS concentration data processed by MNE-Python. Therefore, we cannot determine which dataset is best for the graph algorithm.

#### `FNIRSDataProcessorMNE`

The MNE-Python fNIRS processing is performed using the `FNIRSDataProcessorMNE` class, which accepts the following arguments:

```python
data_dir = '../data/raw'
output_dir = '../data/processed_data_mne'

processor = FNIRSDataProcessorMNE(
    data_dir=data_dir,
    output_dir=output_dir,
    data_type='all',
    task_type='GNG',
    apply_motion_correction=False,
    interpolate_bad_channels=False,
    negative_correlation=False,
    apply_baseline_correction=True,
    apply_zscore=True,
    sci_threshold=0.3,
    save_data_format='npy',
    ppf=1.0
)
processor.run()
```

In initial experiments, this configuration produced fNIRS concentration data with a correlation of approximately `R > 0.9` (tested on one subject, `AH014`) compared to HOMER3 fNIRS concentration data. Therefore, the MNE-Python fNIRS concentration data may be usable for graph algorithms since it has a similar frequency response to the HOMER3 version.

We will conduct further experiments to evaluate the feasibility of creating an end-to-end pipeline from raw data processing to GAD (Generalized Anxiety Disorder) prediction using graph algorithms.

#### `FNIRSDataProcessorHOMER3`

This class is designed to integrate `.CSV` fNIRS concentration data processed by HOMER3 MATLAB into the `MNE-Python` library. The objective is to facilitate the creation of fNIRS data compatible with graph algorithms. Most methods are similar to the MNE version. The key difference is that this class does not perform any signal filtering; it only imports the `.CSV` data and performs additional preprocessing for signal epoching (selecting only neurocognitive tasks: Go/No-Go, 1backWM, VF, and SS), as well as implementing Z-Score Standardization and Baseline Correction.

#### Example: Preprocessed fNIRS Data Folder Structure

```bash
📁 AH021/
├─📄 AH021.data
├─📄 GNG_evoked.png
├─📁 hbo/
│ ├─📄 0.npy
│ └─📄 1.npy
├─📁 hbr/
│ ├─📄 0.npy
│ └─📄 1.npy
└─📁 hbt/
  ├─📄 0.npy
  └─📄 1.npy
```

Each processed dataset (from either `MNE` or `HOMER3` processor) follows the folder structure shown above. Each subject will contain:
- A `.data` file containing metadata such as subject name and sampling frequency
- Concentration folders for `hbo`, `hbr`, and `hbt`
- An `*_evoked.png` file providing a visualization of the average fNIRS concentration data across 4 trials

### `2_fNIRS_graph_algorithm.ipynb`

This notebook is divided into three sections:

#### 1. Data Preparation

This section contains methods and classes for graph data preparation. The `fNIRSGraphDatasetNonRecurrent` class creates a graph dataset object using the [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) library. Inspect the notebook for further details.

#### 2. Graph Model Initialization

This section contains methods and classes for flexible graph algorithms. Currently, the defined model is `FlexibleGAT`, the latest model from our experiments, which achieved the best performance for GAD prediction using graph datasets. It allows flexible graph model initialization without explicitly defining the number of layers, hidden channels, dropout rate, etc. Instead, this model is designed to accept a list of values. Here is an example:

```python
MODEL_CONFIG = {
    'in_channels': 6,  # 6 statistical features (or 6 + PE_WALK_LENGTH if using RWPE)
    'n_layers': 2,
    'n_filters': [112, 32],
    'heads': [6, 4],
    'fc_size': 96,
    'dropout': 0.4,
    'edge_dim': 2,  # correlation + coherence
    'n_classes': 2,
    'use_residual': True,
    'use_norm': True,
    'norm_type': 'batch',
    'use_gine_first_layer': True,
    'gine_train_eps': True
}

def create_model():
    return FlexibleGATNet(**MODEL_CONFIG)
```

This approach facilitates easier experimentation with different hyperparameter combinations.

#### 3. 5-Fold Cross-Validation Training Pipeline

This section contains the training pipeline using 5-Fold Subject Cross-Validation, ensuring that no samples from the training set leak into the validation set. Since each subject completed 4 trials of neurocognitive tasks, it is important to ensure that trials from the training set do not leak into the validation set.
