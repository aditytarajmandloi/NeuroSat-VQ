# NeuroSat VQ

**NeuroSat VQ** is an advanced satellite image compression framework utilizing a Vector Quantized Variational Autoencoder (VQ-VAE). It achieves high compression ratios for geospatial data while maintaining superior perceptual and structural quality.

The system is designed to handle massive satellite imagery (e.g., GeoTIFFs) by processing them in overlapping tiles, compressing them into a compact latent representation using learned vector quantization, and reconstructing them with seamless boundary handling.

## Key Features

- **VQ-VAE Architecture**:
  - **Encoder**: 4-stage ResBlock-based convolutional encoder with SE (Squeeze-and-Excitation) blocks.
  - **Quantizer**: vector quantization with Exponential Moving Average (EMA) updates for stable codebook learning.
  - **Decoder**: PixelShuffle-based upsampling for high-fidelity reconstruction.
- **Geospatial Aware**: Preserves all GeoTIFF metadata (coordinates, projection) via `rasterio`.
- **Loss Functions**:
  - **Charbonnier Loss**: Robust L1-like loss for pixel accuracy.
  - **Sobel Edge Loss**: Preserves high-frequency structural details (roads, buildings).
  - **Focal Frequency Loss**: Ensures spectral fidelity by minimizing errors in the frequency domain (FFT).
- **Tiled Inference**: Handles large images using sliding windows with Hann window blending to eliminate tile boundary artifacts.
- **Compression**: Uses Zlib and bit-packing to store latent indices efficiently.

---

## Directory Structure

```
NeuroSat VQ/
├── data/                   # Dataset directory (e.g., NAIP images)
├── models_v2_7/            # Saved checkpoints
├── src_v2/
│   ├── dataset.py          # Balanced dataloader & class sampling
│   ├── model_v2_7.py       # VQ-VAE model architecture (V2Autoencoder)
│   ├── utils_v2.py         # Loss functions & compression helpers
│   └── overlap_utils.py    # Tiling & grid computation
├── train_v2_7.py           # Main training script with mixed precision
├── compress_v2_7.py        # Inference: Compress GeoTIFF -> .pkl
├── decompress_v2_7.py      # Inference: Decompress .pkl -> GeoTIFF
└── verify_reconstruction.py # Quality Analysis (PSNR, SSIM, Heatmaps)
```

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (Recommended)

### Setup

1.  **Clone the Repository**
    ```bash
    git clone <repo_url>
    cd NeuroSat VQ
    ```

2.  **Install Dependencies**
    ```bash
    pip install torch torchvision numpy scipy dahuffman tqdm Pillow joblib opencv-python scikit-image rasterio
    ```
    *(Note: On Windows, `rasterio` may require installing from a binary wheel if `pip` fails.)*

---

## Usage Guide

### 1. Training

Train the VQ-VAE model on a dataset of satellite images.

**Script**: `train_v2_7.py`

- **Dataset**: Expects images in `data/naip_128` (configurable).
- **Balancing**: Uses `src_v2/dataset.py` to balance sampling across classes (e.g., Urban, Forest, Desert).
- **Mixed Precision**: Uses `torch.amp` (FP16) for faster training.

**Command:**
```bash
python train_v2_7.py
```
*Checkpoints are saved to `models_v2_7/` every 5 epochs. Health check images (`health_<epoch>.png`) are generated to visualize progress.*

### 2. Compression

Compress a large GeoTIFF into a highly compact `.pkl` file.

**Script**: `compress_v2_7.py`

- **Process**:
  1.  Reads the GeoTIFF.
  2.  Pads and tiles the image (128x128 tiles with 16px overlap).
  3.  Encodes tiles into latent indices.
  4.  Packs indices and metadata using Zlib.

**Command:**
```bash
python compress_v2_7.py --input path/to/image.tif --output compressed.pkl --model models_v2_7/v2_7_latest.pth
```
*Output: Displays the compression ratio achieved.*

### 3. Decompression

Reconstruct the original image from the compressed package.

**Script**: `decompress_v2_7.py`

- **Process**:
  1.  Unpacks the latent indices.
  2.  Decodes them back to image tiles using the trained model.
  3.  Stitches tiles together using Hann window blending for smooth edges.
  4.  Restores original GeoTIFF metadata.

**Command:**
```bash
python decompress_v2_7.py --input compressed.pkl --output reconstructed.tif --model models_v2_7/v2_7_latest.pth
```

### 4. Verification

Quantitatively and qualitatively evaluate the reconstruction.

**Script**: `verify_reconstruction.py`

**Metrics Calculated:**
- **PSNR (Global & Patch-wise)**: Peak Signal-to-Noise Ratio.
- **SSIM (Global & Patch-wise)**: Structural Similarity Index.
- **MSE**: Mean Squared Error.
- **Error Heatmap**: Generates `recon_error_heatmap.png` to show where errors occurred.

**Command:**
```bash
python verify_reconstruction.py --orig original.tif --recon reconstructed.tif
```

---

## Technical Details

### Architecture (`src_v2/model_v2_7.py`)
- **Latent Dim**: 96 channels, split into 4 slices for quantization.
- **Codebook**: 4096 embeddings.
- **Downsampling**: 4x reduction (128x128 -> 8x8 latent feature map).

### Training Strategy
- **Optimizer**: AdamW with `lr=5e-5`.
- **Loss Composition**:
  `Total Loss = Charbonnier + VQ_Loss + 0.25 * Edge_Loss + 0.1 * FFT_Loss`
- **Gradient Accumulation**: Steps every 4 batches.
