import numpy as np
import math

def hann_2d(window_size):
    if window_size <= 1:
        return np.ones((1, 1), dtype=np.float32)
    
    w1 = np.hanning(window_size).astype(np.float32)
    w2 = np.hanning(window_size).astype(np.float32)
    return np.outer(w1, w2)

def compute_tile_grid(dim, tile_size, overlap):
    stride = tile_size - overlap
    if dim <= tile_size:
        n_tiles = 1
    else:
        n_tiles = math.ceil((dim - tile_size) / stride) + 1
    
    # Calculate padded size needed to fit the last tile
    padded = max(dim, (n_tiles - 1) * stride + tile_size)
    positions = [i * stride for i in range(n_tiles)]
    
    return positions, padded