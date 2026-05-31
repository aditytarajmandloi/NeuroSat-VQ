# Part 1: Core Architecture, Dataset Pipeline, and Data Flow

## 1. Introduction to NeuroSat VQ v2.7
**NeuroSat VQ** is a state-of-the-art compression framework designed specifically for massive geospatial and satellite imagery (GeoTIFFs). Unlike standard compression algorithms like JPEG—which introduce blocky artifacts and destroy critical geospatial metadata—NeuroSat VQ utilizes a **Vector Quantized Variational Autoencoder (VQ-VAE)**. 

The goal of this project is to take massive, high-resolution satellite tiles, compress them into an incredibly compact binary representation (`.bin`), and decompress them seamlessly while preserving both perceptual quality (buildings, roads, textures) and structural data (GPS coordinates, projections).

---

## 2. The Dataset and Data Pipeline (`src_v2/dataset.py`)
Training a satellite AI requires careful data balancing. If the model only sees empty oceans, it will fail to compress complex cities. The project implements a highly sophisticated data loading pipeline to solve this.

### Balanced Class Sampling
The dataset relies on a predefined `CLASS_SAMPLE_MAP` that dictates exactly how many images to pull from specific biomes. For example:
- `urban`: 6,000 samples (highly complex, needs lots of data)
- `Forest`: 3,500 samples
- `Desert`: 1,000 samples (low complexity)
By balancing the data, the model learns a codebook that is versatile across the entire globe.

### The "Frozen Split" Mechanism
To prevent data leakage between training and testing, the dataloader uses a "Sticky List" feature. When run for the first time, it randomly selects images based on the class map and saves this exact list to `data/train_split_frozen.json`. On subsequent runs, it loads this exact file. This guarantees the model trains on the exact same subset every time, ensuring reproducibility.

### On-the-Fly Cropping
Instead of static images, the `_ListDataset` dynamically generates padded 128x128 crops during training. By applying random padding and cropping, the model implicitly learns how to handle off-center objects and overlapping boundaries, which is crucial for the later seamless-stitching phase.

---

## 3. The Custom Neural Architecture (`src_v2/model_v2_7.py`)
The heart of NeuroSat is the `V2Autoencoder`. It is broken down into three massive components:

### A. The Encoder (Spatial Reduction)
The encoder takes a `128x128` RGB image and compresses it both spatially and depth-wise.
- **Layers:** It uses a 4-stage architecture. Each stage halves the resolution using a stride-2 convolution, reducing the `128x128` image down to an `8x8` feature map.
- **ResBlocks & SE Blocks:** After every convolution, the data passes through a custom Residual Block (`ResBlock`). These blocks use **Squeeze-and-Excitation (SE)** mechanisms. SE blocks act as an "attention" layer, dynamically weighting the importance of different feature channels based on what the network is looking at (e.g., boosting edge-detection channels if it's over a city).

### B. The Latent Space & EMA Vector Quantizer (The Compression)
Once the image is reduced to an `8x8` grid with 96 channels, the magic happens.
- **Slicing:** The 96 channels are sliced into smaller chunks (`SLICE_DIM = 4`). This chunking technique creates a massive number of possible combinations without needing an impossibly large codebook.
- **The Codebook:** The network compares the sliced features to a dictionary of 4,096 learned visual patterns. It throws away the heavy floating-point features and simply keeps the **Integer Index** of the closest pattern. This is what makes the file size so small!
- **Exponential Moving Average (EMA):** Updating 4,096 vectors using standard backpropagation can cause "Codebook Collapse" (where the model only uses 10 patterns and ignores the rest). NeuroSat uses EMA to update the codebook smoothly based on frequency counts, keeping the entire dictionary active and stable.

### C. The Decoder (Reconstruction)
The decoder takes the integer indices, looks up the vectors in the codebook, and rebuilds the `128x128` image.
- **PixelShuffle Upsampling:** Standard AIs use Transposed Convolutions to scale up, which creates ugly "checkerboard" patterns. NeuroSat instead uses PyTorch's `PixelShuffle` (`nn.PixelShuffle(2)`), which rearranges channels into spatial pixels. This produces much sharper, cleaner reconstructions of hard edges like roads and rooftops.

*In Part 2, we will explore exactly how this massive architecture is trained using custom loss functions, and how the overlapping tiling math works to compress infinite-scale GeoTIFFs.*
