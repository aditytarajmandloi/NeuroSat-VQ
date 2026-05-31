import os
import sys
import time
import base64
import subprocess
from PIL import Image
import numpy as np

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Allow uploads up to 500 MB
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# Folders
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'data', 'api_uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'data', 'api_outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Model
MODEL_PATH = os.environ.get('NEUROSAT_MODEL_PATH', 'models_v2_7/v2_7_finalmodel.pth')
MOCK_PROCESSING = False

ALLOWED_EXTENSIONS_COMPRESS = {'png', 'jpg', 'jpeg', 'tiff', 'tif'}
ALLOWED_EXTENSIONS_DECOMPRESS = {'bin'}

def allowed_file(filename, allowed_set):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set

# ─── Routes ───
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "NeuroSaT API", "version": "v2.7"}), 200

@app.route('/api/v1/compress', methods=['POST'])
def compress_image():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    if not allowed_file(file.filename, ALLOWED_EXTENSIONS_COMPRESS):
        return jsonify({"status": "error", "message": f"File type not supported. Allowed: {ALLOWED_EXTENSIONS_COMPRESS}"}), 400

    filename = secure_filename(file.filename)
    input_filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_filepath)

    basename = os.path.splitext(filename)[0]
    output_bin_name = f"{basename}.bin"
    output_filepath = os.path.join(OUTPUT_FOLDER, output_bin_name)

    try:
        start = time.time()
        
        cmd = [
            sys.executable, "compress_v2_7.py",
            "-i", input_filepath,
            "-o", output_filepath,
            "-m", MODEL_PATH
        ]
        
        print("Running:", " ".join(cmd))
        
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=env)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
        if result.returncode != 0 or not os.path.exists(output_filepath):
            return jsonify({
                "status": "error", 
                "message": f"Compression script failed. Exit code {result.returncode}",
                "stderr": result.stderr
            }), 500

        elapsed = int((time.time() - start) * 1000)

        original_size = os.path.getsize(input_filepath)
        compressed_size = os.path.getsize(output_filepath)
        ratio = f"{(original_size / compressed_size):.2f}x" if compressed_size > 0 else "N/A"

        return jsonify({
            "status": "success",
            "message": "Image compressed successfully.",
            "metrics": {
                "original_size_bytes": original_size,
                "compressed_size_bytes": compressed_size,
                "compression_ratio": ratio,
                "time_ms": elapsed
            },
            "download_url": f"/api/v1/downloads/{output_bin_name}"
        }), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/v1/decompress', methods=['POST'])
def decompress_bin():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    if not allowed_file(file.filename, ALLOWED_EXTENSIONS_DECOMPRESS):
        return jsonify({"status": "error", "message": "Must upload a .bin file"}), 400

    filename = secure_filename(file.filename)
    input_filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_filepath)

    basename = os.path.splitext(filename)[0]
    output_img_name = f"{basename}_reconstructed.png"
    output_filepath = os.path.join(OUTPUT_FOLDER, output_img_name)

    # For TIFF compatibility with frontend (frontend expects PNG)
    output_png_path = os.path.join(OUTPUT_FOLDER, f"{basename}_reconstructed.png")
    
    # Actually wait the script saves to output_filepath. Let's make it explicitly save as .tif or whatever
    # The script uses rasterio so it probably saves a TIF file by default depending on profile
    # Let's see - decompress_v2_7.py writes out using `with rasterio.open(args.output, "w", **prof) as dst`.
    # It takes the profile from the original image (which was saved in the header if modified... wait, did the new script save the profile?)
    # Before we modified it, the script used to read. Let's make output_filepath a .tif here to be safe since rasterio is writing it.
    output_tif_path = os.path.join(OUTPUT_FOLDER, f"{basename}_reconstructed.tif")

    try:
        start = time.time()
        
        cmd = [
            sys.executable, "decompress_v2_7.py",
            "-i", input_filepath,
            "-o", output_tif_path,
            "-m", MODEL_PATH
        ]
        
        print("Running:", " ".join(cmd))
        
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=env)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        if result.returncode != 0 or not os.path.exists(output_tif_path):
            return jsonify({
                "status": "error", 
                "message": f"Decompression script failed. Exit code {result.returncode}",
                "stderr": result.stderr
            }), 500
            
        elapsed = int((time.time() - start) * 1000)

        # Convert the rasterio output (typically TIF) to PNG for the frontend base64 image display
        import rasterio
        with rasterio.open(output_tif_path) as src:
            img_data = src.read().transpose(1, 2, 0)
        
        # Save as PNG
        Image.fromarray(img_data).save(output_filepath)

        with open(output_filepath, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')

        return jsonify({
            "status": "success",
            "message": "Image reconstructed successfully.",
            "metrics": {"reconstruction_time_ms": elapsed},
            "image_data": img_b64,
            "mime_type": "image/png"
        }), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/v1/downloads/<filename>', methods=['GET'])
def download_file(filename):
    filepath = os.path.join(OUTPUT_FOLDER, secure_filename(filename))
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({"status": "error", "message": "File not found"}), 404


@app.route('/api/v1/verify', methods=['POST'])
def verify_reconstruction():
    if 'original' not in request.files or 'reconstructed' not in request.files:
        return jsonify({"status": "error", "message": "Both 'original' and 'reconstructed' files are required"}), 400

    orig_file = request.files['original']
    recon_file = request.files['reconstructed']

    if orig_file.filename == '' or recon_file.filename == '':
        return jsonify({"status": "error", "message": "Both files must be selected"}), 400

    orig_name = secure_filename(orig_file.filename)
    recon_name = secure_filename(recon_file.filename)
    orig_path = os.path.join(UPLOAD_FOLDER, f"verify_orig_{orig_name}")
    recon_path = os.path.join(UPLOAD_FOLDER, f"verify_recon_{recon_name}")
    orig_file.save(orig_path)
    recon_file.save(recon_path)

    try:
        from verify_reconstruction import global_metrics, patch_metrics, error_heatmap

        orig = np.array(Image.open(orig_path).convert("RGB"))
        recon = np.array(Image.open(recon_path).convert("RGB"))

        if orig.shape != recon.shape:
            return jsonify({
                "status": "error",
                "message": f"Image dimensions don't match: original {orig.shape[:2]} vs reconstructed {recon.shape[:2]}"
            }), 400

        mse, psnr, ssim = global_metrics(orig, recon)

        # Adaptive patch size for small images
        h, w = orig.shape[:2]
        min_dim = min(h, w)
        try:
            if min_dim >= 256:
                p_psnr_mean, p_psnr_min, p_ssim_mean, p_ssim_min = patch_metrics(orig, recon, patch=256, stride=128)
            elif min_dim >= 64:
                p_psnr_mean, p_psnr_min, p_ssim_mean, p_ssim_min = patch_metrics(orig, recon, patch=min_dim, stride=max(min_dim // 2, 8))
            else:
                p_psnr_mean = p_psnr_min = psnr
                p_ssim_mean = p_ssim_min = ssim
        except Exception:
            p_psnr_mean = p_psnr_min = psnr
            p_ssim_mean = p_ssim_min = ssim

        heat = error_heatmap(orig, recon)
        heat_rgb = heat[:, :, ::-1]  # BGR → RGB for PIL
        heatmap_path = os.path.join(OUTPUT_FOLDER, "verify_heatmap.png")
        Image.fromarray(heat_rgb).save(heatmap_path)

        with open(heatmap_path, "rb") as f:
            heatmap_b64 = base64.b64encode(f.read()).decode('utf-8')

        return jsonify({
            "status": "success",
            "metrics": {
                "mse": round(float(mse), 6),
                "psnr": round(float(psnr), 2),
                "ssim": round(float(ssim), 4),
                "patch_psnr_mean": round(float(p_psnr_mean), 2),
                "patch_psnr_min": round(float(p_psnr_min), 2),
                "patch_ssim_mean": round(float(p_ssim_mean), 4),
                "patch_ssim_min": round(float(p_ssim_min), 4),
                "resolution": f"{orig.shape[1]}x{orig.shape[0]}"
            },
            "heatmap_data": heatmap_b64,
            "heatmap_mime": "image/png"
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    print(">>> Starting NeuroSaT VQ Backend API...")
    app.run(host='0.0.0.0', port=5000, debug=False)
