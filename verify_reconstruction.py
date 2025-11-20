import argparse
import numpy as np
from PIL import Image
import warnings
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

import skimage.metrics as m
import cv2


# ================================
# GLOBAL METRICS (FULLY FIXED)
# ================================
def global_metrics(orig, recon):

    orig_f = orig.astype(np.float32) / 255.0
    recon_f = recon.astype(np.float32) / 255.0

    mse = np.mean((orig_f - recon_f)**2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 100.0

    # -------- FIXED SSIM CALL ----------
    s = m.structural_similarity(
        orig_f,
        recon_f,
        channel_axis=-1,          # NEW API for skimage 0.19+
        win_size=7,
        gaussian_weights=True,
        sigma=1.5,
        data_range=1.0            # REQUIRED for float input
    )
    # -----------------------------------

    return mse, psnr, s


# ================================
# PATCH METRICS
# ================================
def patch_metrics(orig, recon, patch=256, stride=128):
    H, W, _ = orig.shape

    psnrs = []
    ssims = []

    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):

            o = orig[y:y+patch, x:x+patch].astype(np.float32) / 255.0
            r = recon[y:y+patch, x:x+patch].astype(np.float32) / 255.0

            mse = np.mean((o - r)**2)
            psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 100.0
            psnrs.append(psnr)

            ssim = m.structural_similarity(
                o,
                r,
                channel_axis=-1,
                win_size=7,
                gaussian_weights=True,
                sigma=1.5,
                data_range=1.0
            )
            ssims.append(ssim)

    return np.mean(psnrs), np.min(psnrs), np.mean(ssims), np.min(ssims)


# ================================
# HEATMAP
# ================================
def error_heatmap(orig, recon):
    diff = np.mean((orig.astype(np.float32) - recon.astype(np.float32))**2, axis=2)
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    return heat


# ================================
# MAIN
# ================================
def evaluate(orig_path, recon_path):

    print("\nLoading images...")
    orig = np.array(Image.open(orig_path).convert("RGB"))
    recon = np.array(Image.open(recon_path).convert("RGB"))

    print("\n========== GLOBAL METRICS ==========")
    mse, psnr, ssim = global_metrics(orig, recon)
    print(f"MSE:  {mse:.4f}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")

    print("\n========== PATCH METRICS ==========")
    pmean, pmin, smean, smin = patch_metrics(orig, recon)
    print(f"Patch PSNR Mean: {pmean:.2f} dB")
    print(f"Patch PSNR Min:  {pmin:.2f} dB")
    print(f"Patch SSIM Mean: {smean:.4f}")
    print(f"Patch SSIM Min:  {smin:.4f}")

    print("\nSaving heatmap -> recon_error_heatmap.png")
    heat = error_heatmap(orig, recon)
    Image.fromarray(heat).save("recon_error_heatmap.png")
    print("Done.")


# ================================
# ENTRY POINT
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig", required=True)
    parser.add_argument("--recon", required=True)
    args = parser.parse_args()

    evaluate(args.orig, args.recon)
    