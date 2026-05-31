# Part 2: Training Pipeline, Custom Loss Functions, and Overlap Mathematics

Following the dataset preparation and architecture definition in Part 1, Part 2 explores how the NeuroSat VQ engine actually learns, how it is penalized for mistakes, and how the mathematics of tiling allow it to handle massive GeoTIFFs without memory crashes or visible seams.

---

## 1. The Training Strategy (`train_v2_7.py`)
Training a VQ-VAE on high-resolution satellite imagery is computationally expensive and notoriously unstable. The training script implements several enterprise-grade techniques to ensure stable convergence on consumer GPUs.

### Automatic Mixed Precision (AMP)
To save GPU Memory (VRAM) and drastically speed up the training loop, the script uses PyTorch's `autocast` and `GradScaler`. 
- Forward passes are executed in **FP16** (16-bit floating point).
- The `GradScaler` prevents gradients from "underflowing" (becoming zero due to the limited precision of 16-bit floats) by dynamically scaling them before the backward pass.

### Simulated Batching (Gradient Accumulation)
Because satellite tiles take up immense memory, the maximum batch size that fits in VRAM might only be 16. The script uses an `ACCUM = 4` parameter. It runs 4 forward and backward passes without updating the model, accumulating the gradients. On the 4th step, it updates the weights (`scaler.step(opt)`). This effectively simulates a batch size of 64, leading to much smoother and more accurate gradient descents.

### Stability Guards
- **NaN Guards:** If an explosive gradient causes the loss to become `NaN` (Not a Number) or infinite, the script immediately intercepts it (`torch.isfinite(loss)`), zeroes the gradients, and skips the batch entirely, preventing the model weights from being instantly corrupted.
- **Gradient Clipping:** Gradients are strictly clipped at a magnitude of `1.0` (`torch.nn.utils.clip_grad_norm_`) to prevent sudden massive updates (Codebook collapse).
- **Automated Health Checks:** Every 5 epochs, the training loop pauses to run a static `health_check.png` through the model. It saves a visual output (`health_<epoch>.png`), allowing developers to visually track the reconstruction quality over time.

---

## 2. The Custom Loss Composition (`src_v2/utils_v2.py`)
Standard neural networks use MSE (Mean Squared Error or L2 Loss). MSE heavily penalizes large errors (squaring them) and ignores tiny errors, which causes image models to output blurry, "safe" averages. Satellite imagery requires absolute sharpness. NeuroSat solves this using a custom multi-objective loss function:

### A. Charbonnier Loss (Robust L1)
Instead of MSE, the baseline color accuracy is enforced by Charbonnier Loss: `sqrt((pred - target)^2 + epsilon)`. 
It acts like an L1 (Absolute Error) loss but is differentiable everywhere. It does not over-penalize outliers, resulting in far less blurring than MSE.

### B. Sobel Edge Loss (Weighted at 0.25)
Satellite imagery is defined by rigid structures: building footprints, roads, and property lines. 
The script dynamically generates Horizontal and Vertical **Sobel filter kernels** and convolves them over both the predicted and target images. It calculates the magnitude of the gradients (edges) and applies an L1 loss between them. This violently forces the AI to align its generated boundaries exactly with the original image.

### C. Focal Frequency (FFT) Loss (Weighted at 0.1)
Neural networks notoriously struggle to recreate high-frequency noise (e.g., the grainy texture of grass, sand, or asphalt), often producing surfaces that look "painted" or "waxy". 
The script uses a Fast Fourier Transform (`torch.fft.fft2`) to convert the images from the spatial pixel domain into the frequency domain. It separates the real and imaginary components and calculates the distance. This loss strictly enforces spectral and textural fidelity.

---

## 3. Data Handling: Overlap Tiling Mathematics (`src_v2/overlap_utils.py`)
A single GeoTIFF can be 10,000 x 10,000 pixels. No GPU can process this at once. The image must be tiled.

### Grid Computation
The `compute_tile_grid` function mathematically divides the massive image into `128x128` tiles. Crucially, it enforces a strict `16-pixel` overlap between all adjacent tiles. It also calculates exactly how much zero-padding is required on the bottom and right edges to ensure the final tiles perfectly fit the 128x128 requirement without cutting off data.

### Hann Window Blending (Seamless Stitching)
If you compress tiles individually and place them side-by-side to decompress, you will get terrible, highly visible grid-lines (seams) due to the neural network treating the edges of the tile differently than the center.
NeuroSat completely eliminates seams using a **2D Hann Window** (`hann_2d`).
- The Hann window is a mathematical mask where the center pixels equal `1.0` (fully opaque) and smoothly taper off to `0.0` (fully transparent) at the edges.
- During decompression (`decompress_v2_7.py`), every reconstructed tile is multiplied by this Hann window.
- Because the tiles were generated with a 16-pixel overlap, the faded edges of two adjacent tiles are mathematically added together. 
- The sum of overlapping Hann windows equals exactly `1.0`. The transition between tiles becomes perfectly smooth, rendering the tile boundaries mathematically invisible.

*In Part 3, we will explore how this AI core is exposed to the user via a robust Flask API, packaged with Zlib, and evaluated using quantitative scripts.*
