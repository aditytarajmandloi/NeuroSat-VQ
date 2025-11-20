import os
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import warnings

# --- Config ---
# Point this to your CLASS FOLDERS (e.g., raw_data/1_Urban)
SOURCE_DIR = "data/New folder" 
# This is our NEW structured output
OUTPUT_DIR = "data/" 
TARGET_SIZE = 128

# --- S-GRADE ADAPTIVE "BORING" FILTER ---
# A higher number = "be stricter" = discard more.
# YOU MUST EDIT THIS TO MATCH YOUR FOLDER NAMES!
BORING_THRESHOLDS = {
    "__default__": 10,          # Default for complex classes
    "9_Flat": 30.0,               # Example: High threshold
    "10_Water": 25.0,             # Example: High threshold
    "salt_utha": 0,            # Your class
}
# --- --- --- ---

def main():
    print(f"Starting S-Grade Preprocessing (v7 - Adaptive Sorter)...")
    print(f"Reading class folders from: {SOURCE_DIR}")
    
    class_folders = [f for f in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, f))]
    
    if not class_folders:
        print(f"Error: No class folders found in {SOURCE_DIR}.")
        print("Please structure your data as 'raw_data/1_Urban/', etc.")
        return

    total_tiles_created = 0
    total_tiles_discarded = 0
    
    Image.MAX_IMAGE_PIXELS = None
    warnings.simplefilter('ignore', Image.DecompressionBombWarning)

    for class_name in class_folders:
        class_source_dir = os.path.join(SOURCE_DIR, class_name)
        class_output_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        
        current_threshold = BORING_THRESHOLDS.get(class_name, BORING_THRESHOLDS["__default__"])
        
        tif_paths = glob(os.path.join(class_source_dir, "*.tif"))
        
        if not tif_paths:
            print(f"Warning: No .tif files found in {class_source_dir}. Skipping.")
            continue
            
        print(f"\nProcessing Class: '{class_name}' ({len(tif_paths)} TIF files)")
        print(f"  Using adaptive 'boring' threshold: {current_threshold}")

        for tif_path in tif_paths:
            print(f"  Tiling file: {os.path.basename(tif_path)}")
            try:
                with Image.open(tif_path) as img:
                    
                    if img.mode == 'RGBA' or len(img.getbands()) == 4:
                        bands = img.split()
                        img = Image.merge("RGB", (bands[0], bands[1], bands[2]))
                    elif img.mode != 'RGB':
                        print(f"    Skipping {os.path.basename(tif_path)}: Unsupported mode {img.mode}")
                        continue
                        
                    width, height = img.size
                    grid_w = width // TARGET_SIZE
                    grid_h = height // TARGET_SIZE
                    
                    for j in tqdm(range(grid_h)):
                        for i in range(grid_w):
                            left, top = i * TARGET_SIZE, j * TARGET_SIZE
                            right, bottom = left + TARGET_SIZE, top + TARGET_SIZE
                            
                            tile = img.crop((left, top, right, bottom))
                            
                            tile_data = np.array(tile)
                            if tile_data.std() < current_threshold:
                                total_tiles_discarded += 1
                                continue 
                            
                            out_path = os.path.join(class_output_dir, f"tile_{total_tiles_created:07d}.png")
                            tile.save(out_path, "PNG")
                            total_tiles_created += 1
                            
            except Exception as e:
                print(f"    Error processing {tif_path}: {e}")

    print(f"\n✅ S-Grade (ADAPTIVE) Preprocessing complete.")
    print(f"   Total useful tiles created: {total_tiles_created} (in {len(class_folders)} classes)")
    print(f"   Boring tiles discarded:     {total_tiles_discarded}")

if __name__ == "__main__":
    main()