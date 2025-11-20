# ===========================
# train_v2_7.py
# ===========================
import os
import torch
from torch import optim
from torch.amp import autocast, GradScaler
import numpy as np
from PIL import Image

from src_v2.dataset import get_balanced_dataloader
from src_v2.model_v2_7 import V2Autoencoder
from src_v2.utils_v2 import (
    charbonnier_loss, sobel_loss, focal_frequency_loss
)

DATA_DIR = "data/naip_128"
MODELS_DIR = "models_v2_7"
MODEL_LATEST = os.path.join(MODELS_DIR, "v2_7_latest.pth")
BATCH_SIZE = 16
ACCUM = 4
EPOCHS = 120

LR = 5e-5


# -------------------------
# PSNR & Health Check
# -------------------------
def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))


def safe_health_check(model, device, epoch):
    img_path = "health_check.png"
    if not os.path.exists(img_path):
        print("⚠️ No health image.")
        return

    img = np.array(Image.open(img_path))/255
    
    # ------------------ FIX IS HERE ------------------
    # Added .float() to force Float32 before sending to GPU
    t = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0).to(device)
    # -------------------------------------------------

    with torch.no_grad(), autocast("cuda"):
        out, _, _ = model(t)

    out_np = out[0].permute(1,2,0).cpu().numpy()
    Image.fromarray((out_np*255).astype(np.uint8)).save(f"health_{epoch}.png")
    print("Saved health image.")


# -------------------------
# Training
# -------------------------
def train():
    device = torch.device("cuda")
    os.makedirs(MODELS_DIR, exist_ok=True)

    loader = get_balanced_dataloader(DATA_DIR, batch_size=BATCH_SIZE,
                                     num_workers=2)
    if loader is None:
        return

    model = V2Autoencoder().to(device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = GradScaler()  # FIXED

    start = 1
    if os.path.exists(MODEL_LATEST):
        ckpt = torch.load(MODEL_LATEST, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        start = ckpt["epoch"]+1
        print(f"Resumed from Epoch {start}")

        # Hard-Reset EMA (optional)
        model.quantizer.reset_ema()

    print("Training V2.7...")

    for epoch in range(start, EPOCHS+1):
        print(f"\nEpoch {epoch} — LR={LR}")
        model.train()
        running = []

        opt.zero_grad()

        for i, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device)

            with autocast("cuda"):
                out, vq_loss, _ = model(imgs)

                l_char = charbonnier_loss(out.float(), imgs.float())
                l_edge = sobel_loss(out.float(), imgs.float()) * 0.25
                l_fft  = focal_frequency_loss(out.float(), imgs.float()) * 0.1

                loss = (l_char + vq_loss + l_edge + l_fft) / ACCUM

            # -------- NaN guard --------
            if not torch.isfinite(loss):
                print(f"[NaN] Epoch {epoch}, Step {i}. Skipping batch.")
                opt.zero_grad(set_to_none=True)
                scaler.update()
                continue

            scaler.scale(loss).backward()

            if (i+1) % ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            running.append(loss.item() * ACCUM)

            if i % 200 == 0:
                print(f"Step {i}/{len(loader)} | Loss={running[-1]:.4f}")

        avg_loss = sum(running)/len(running)
        print(f"Epoch {epoch} | Avg Loss={avg_loss:.5f}")

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict()
        }, MODEL_LATEST)

        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict()
            }, os.path.join(MODELS_DIR, f"v2_7_e{epoch}.pth"))

            safe_health_check(model, device, epoch)


if __name__ == "__main__":
    train()
