import os
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import warnings

# --- Config ---
# Point this to your data folder that is FULL of tiles
DATA_DIR = "data/naip_stratified_128/New folder" 

# --- S-GRADE ADAPTIVE "BORING" FILTER ---
# This is the "S-Grade" part.
# A higher number = "be stricter" = discard more.
# You MUST edit this to match your folder names.
BORING_THRESHOLDS = {
    "__default__": 20,          # Default for complex classes (Urban, Forest, etc.)
    "salt_utha": 40.0,            # YOUR CLASS: High threshold to discard boring salt
    "9_Flat": 30.0,               # Example: High threshold for "Flat"
    "10_Water": 25.0,             # Example: High threshold for "Water"
    # Add your other "boring" class folders here    
}
# --- --- --- ---

def main():
    print(f"--- Starting S-Grade CURATION Script (The Fast Cleaner) ---")
    print(f"Scanning data folders in: {DATA_DIR}")
    
    class_folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]
    
    if not class_folders:
        print(f"Error: No class folders found in {DATA_DIR}.")
        return

    total_tiles_kept = 0
    total_tiles_discarded = 0
    
    Image.MAX_IMAGE_PIXELS = None
    warnings.simplefilter('ignore', Image.DecompressionBombWarning)

    # --- Loop 1: Iterate over each CLASS folder ---
    for class_name in class_folders:
        class_dir = os.path.join(DATA_DIR, class_name)
        
        # Get the specific "boring" threshold for this class
        current_threshold = BORING_THRESHOLDS.get(class_name, BORING_THRESHOLDS["__default__"])
        
        # Find all the .png tiles in this folder
        tile_paths = glob(os.path.join(class_dir, "*.png"))
        
        if not tile_paths:
            print(f"Warning: No .png tiles found in {class_dir}. Skipping.")
            continue
            
        print(f"\nCurating Class: '{class_name}' ({len(tile_paths)} tiles)")
        print(f"  Using adaptive 'boring' threshold: {current_threshold}")

        # --- Loop 2: Iterate over each .png TILE in that class ---
        for tile_path in tqdm(tile_paths, desc=f"Cleaning {class_name}"):
            try:
                with Image.open(tile_path) as tile:
                    # --- S-GRADE ADAPTIVE CHECK ---
                    # Convert to numpy array and check Standard Deviation
                    tile_data = np.array(tile)
                    if tile_data.std() < current_threshold:
                        # This tile is "boring".
                        total_tiles_discarded += 1
                        # We must close the file handle *before* deleting!
                        tile.close() 
                        os.remove(tile_path) # Delete the file
                    else:
                        # This tile is "useful". Keep it.
                        total_tiles_kept += 1
                    # --- End Check ---
                            
            except Exception as e:
                print(f"    Error processing {tile_path}: {e}")

    print(f"\n✅ S-Grade Curation complete.")
    print(f"   Total useful tiles kept:  {total_tiles_kept}")
    print(f"   Boring tiles discarded: {total_tiles_discarded}")

if __name__ == "__main__":
    main()