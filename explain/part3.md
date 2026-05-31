# Part 3: Inference, Data Packing, Verification, and the Web API

This final section explores how the trained AI model is deployed for inference. We will trace the journey of a large GeoTIFF as it is ingested, compressed into a binary file, verified for scientific accuracy, and served via a robust Web API.

---

## 1. The Compression Pipeline (`compress_v2_7.py`)
When a user uploads an image, the script executes a highly optimized inference loop to generate the `.bin` package.

### Geographic Metadata Extraction
Standard AI tools destroy metadata. NeuroSat uses the `rasterio` library to open the GeoTIFF. Before passing pixels to the AI, it explicitly extracts the `geo_profile`—a dictionary containing the Coordinate Reference System (CRS), GPS bounding box, and affine transformations. This ensures the output will still be a valid map.

### Inference and Alpha Handling
- To prevent Out-Of-Memory (OOM) crashes, if an image has an Alpha (transparency) channel, the script isolates it. In earlier versions, this Alpha channel was compressed losslessly using Run-Length Encoding (RLE) to save space. 
- The RGB image is padded and split into tiles. Each tile is passed through the model. But instead of generating an image, the script captures the **Latent Indices** (the integer addresses from the Codebook).

### Zlib Bit-Packing (`utils_v2.py`)
After the entire image is processed, the system has a massive 1D array of integers. 
The `pack_indices` function converts this PyTorch tensor into a highly efficient `numpy.uint16` array. It then converts the array to raw bytes and compresses it using `zlib` at maximum compression (`level=9`). This lossless compression step squeezes the data down by another 20-40% before it is saved alongside the `geo_profile` into a Python `pickle` payload (`.bin`).

---

## 2. The Decompression Pipeline (`decompress_v2_7.py`)
Reconstructing the image is the reverse of compression, with specific mathematical safeguards.

### Functional Embedding Lookup
The script un-pickles the file and decompresses the `zlib` blob back into integers. Instead of passing these integers into a standard layer, the script uses PyTorch's `F.embedding()` functional API to look up the exact visual features from the Codebook. This approach is memory-safe and prevents accidental backpropagation or state tracking during inference.

### Hann Window Accumulation
As the decoder scales the features back up to `128x128` tiles, the script creates two accumulation arrays: `acc` (for the pixel data) and `weights` (to track overlap).
Every reconstructed tile is multiplied by the 2D Hann Window (discussed in Part 2). The pixels are added to `acc`, and the window values are added to `weights`. Finally, the script divides the accumulated pixels by the weights (`final_rgb = acc / weights`). This perfectly blends the overlapping tiles together. 
The original `rasterio` profile is re-attached, and the final GeoTIFF is written to disk.

---

## 3. Scientific Quality Verification (`verify_reconstruction.py`)
Because satellite imagery is often used for agriculture, military, or legal applications, the system must mathematically prove that the reconstruction is faithful.

### Global vs. Patch Metrics
The script uses `scikit-image` to calculate **PSNR** (Peak Signal-to-Noise Ratio) and **SSIM** (Structural Similarity Index). 
However, global metrics can hide terrible localized errors (e.g., the image is 99% ocean, but the 1% that is a city was completely blurred). To solve this, the script runs a **Patch-wise Evaluation**. It slides a `256x256` window across the image, calculating the SSIM of every patch, and reports the *Minimum Patch SSIM*. This guarantees the user knows the absolute worst-case error anywhere on the map.

### Error Heatmaps
The script generates a visual heatmap (`recon_error_heatmap.png`). It calculates the Mean Squared Error for every single pixel, normalizes it, and applies OpenCV's `COLORMAP_JET`. Areas of high error (often busy intersections or power lines) glow bright red, allowing researchers to instantly see where the AI struggled.

---

## 4. The Orchestration API (`app.py`)
To make this complex pipeline accessible via the React/Vite Web UI, the backend uses a **Flask API**.

- **Isolated Subprocessing:** Running heavy PyTorch inference directly inside a Flask route is dangerous—it leaks memory and can crash the web server. Instead, the API acts as an orchestrator. It uses `subprocess.run()` to execute the `compress` and `decompress` Python scripts in entirely separate, isolated background processes.
- **Large File Support:** The API configuration (`MAX_CONTENT_LENGTH`) is explicitly set to `500 MB` to accommodate massive TIFF uploads without triggering HTTP 413 (Payload Too Large) errors.
- **Base64 Previews:** When the `/api/v1/decompress` route is hit, it doesn't just save a file. It converts the reconstructed image into a Base64 string. This allows the frontend to instantly render the image on the user's screen without forcing them to download a secondary file.

*This concludes the deep dive into the NeuroSat VQ architecture! From balanced data loading to vector quantization, semantic overlap blending, and robust API orchestration, the pipeline is a complete, production-ready solution for geospatial compression.*
