import torch
import torch.nn.functional as F
import torch.fft
import zlib
import numpy as np

def charbonnier_loss(pred, target, eps=1e-6):
    """
    Charbonnier Loss (Robust L1).
    Formula: sqrt((x-y)^2 + epsilon)
    Why: MSE (L2) squares errors, which over-punishes outliers and leads to blurring.
    L1 is sharper, and Charbonnier is a differentiable version of L1.
    """
    diff = pred - target
    loss = torch.mean(torch.sqrt(diff * diff + eps))
    return loss

def sobel_loss(pred, target):
    """
    Sobel Edge Loss.
    Why: Satellite images need sharp roads and buildings.
    We run a Sobel filter (edge detector) on both the Prediction and Target.
    We then calculate the error between the *Edges*, not just the colors.
    This forces the model to align boundaries perfectly.
    """
    # FIX: Use dynamic dtype to match input (Float16/32) to prevent crashes
    dtype = pred.dtype
    device = pred.device
    
    # Horizontal Edge Kernel
    fx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    # Vertical Edge Kernel
    fy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    
    total_loss = 0
    # Apply independently to R, G, B channels
    for c in range(3):
        p_c = pred[:, c:c+1]
        t_c = target[:, c:c+1]
        
        p_x = F.conv2d(p_c, fx, padding=1)
        p_y = F.conv2d(p_c, fy, padding=1)
        p_edge = torch.sqrt(p_x**2 + p_y**2 + 1e-6) # Gradient Magnitude
        
        t_x = F.conv2d(t_c, fx, padding=1)
        t_y = F.conv2d(t_c, fy, padding=1)
        t_edge = torch.sqrt(t_x**2 + t_y**2 + 1e-6)
        
        total_loss += F.l1_loss(p_edge, t_edge)
    return total_loss

def focal_frequency_loss(pred, target):
    """
    FFT (Fast Fourier Transform) Loss.
    Why: Neural Networks often struggle with high-frequency texture (grain, asphalt).
    We convert the image to the Frequency Domain and calculate loss there.
    This screams at the model if it produces a "waxy" or overly smooth image.
    """
    # FIX: Force Float32. FFT is unstable in Float16 (AMP).
    pred = pred.float()
    target = target.float()
    
    # Convert spatial (H,W) to frequency (H,W)
    pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
    target_fft = torch.fft.fft2(target, dim=(-2, -1))
    
    # Separate Real and Imaginary components
    pred_fft = torch.stack([pred_fft.real, pred_fft.imag], -1)
    target_fft = torch.stack([target_fft.real, target_fft.imag], -1)
    
    # Calculate difference magnitude
    diff = pred_fft - target_fft
    return torch.mean(torch.sqrt(diff.pow(2).sum(-1) + 1e-6))

# --- COMPRESSION HELPERS ---

def pack_indices(indices_tensor):
    # Zlib compression for the integer indices (Lossless)
    if torch.is_tensor(indices_tensor):
        arr = indices_tensor.detach().cpu().numpy().astype(np.uint16)
    else:
        arr = indices_tensor.astype(np.uint16)
    return zlib.compress(arr.tobytes(), level=9)

def unpack_indices(compressed_bytes, count, dtype=np.uint16):
    raw_bytes = zlib.decompress(compressed_bytes)
    return np.frombuffer(raw_bytes, dtype=dtype)

def rle_encode(mask):
    """Run-Length Encode for Alpha Mask (Compresses '0 0 0 0' to '4x0')"""
    pixels = mask.flatten()
    changes = np.concatenate(([0], np.where(pixels[:-1] != pixels[1:])[0] + 1, [len(pixels)]))
    lengths = np.diff(changes)
    values = pixels[changes[:-1]]
    encoded = []
    for l, v in zip(lengths, values):
        encoded.append((l, v))
    return encoded

def rle_decode(encoded_list, shape):
    pixels = []
    for length, value in encoded_list:
        pixels.extend([value] * length)
    return np.array(pixels, dtype=np.uint8).reshape(shape)