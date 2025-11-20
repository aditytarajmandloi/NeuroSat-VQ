import os
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import warnings

# --- Config ---
# This is the folder with your CLASS subfolders (e.g., raw_data/1_Urban)
SOURCE_DIR = "data/New folder" 
# --- --- --- ---

def main():
    print(f"--- Starting S-Grade RECURSIVE Quadrant Splitter (v9) ---")
    print(f"Scanning class folders in: {SOURCE_DIR}")
    
    class_folders = [f for f in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, f))]
    
    if not class_folders:
        print(f"Error: No class folders found in {SOURCE_DIR}.")
        return

    Image.MAX_IMAGE_PIXELS = None
    warnings.simplefilter('ignore', Image.DecompressionBombWarning)

    total_quads_kept = 0
    total_files_split = 0

    # --- Loop 1: Iterate over each CLASS folder (e.g., "1_Urban") ---
    for class_name in class_folders:
        class_source_dir = os.path.join(SOURCE_DIR, class_name)
        
        # Find all .tif files *inside* this class folder
        # *** S-GRADE FIX (v9) ***
        # We now find ALL .tif files, with no filter.
        tif_paths = glob(os.path.join(class_source_dir, "*.tif"))
        
        if not tif_paths:
            print(f"\nNo .tif files found in {class_source_dir}. Skipping.")
            continue
            
        print(f"\nProcessing Class: '{class_name}' ({len(tif_paths)} TIF files)")
        
        # --- Loop 2: Iterate over each GIANT TIF in that class ---
        for tif_path in tif_paths:
            print(f"  Splitting: {os.path.basename(tif_path)}")
            total_files_split += 1
            try:
                with Image.open(tif_path) as img:
                    
                    # --- 4-Band to 3-Band Fix ---
                    if img.mode == 'RGBA' or len(img.getbands()) == 4:
                        bands = img.split()
                        img = Image.merge("RGB", (bands[0], bands[1], bands[2]))
                    elif img.mode != 'RGB':
                        print(f"    Skipping: Unsupported mode {img.mode}")
                        continue
                    # --- End Fix ---
                        
                    width, height = img.size
                    mid_x = width // 2
                    mid_y = height // 2
                    
                    quad_boxes = {
                        "_Q1_top_left": (0, 0, mid_x, mid_y),
                        "_Q2_top_right": (mid_x, 0, width, mid_y),
                        "_Q3_bottom_left": (0, mid_y, mid_x, height),
                        "_Q4_bottom_right": (mid_x, mid_y, width, height)
                    }
                    
                    base_filename = os.path.splitext(os.path.basename(tif_path))[0]
                    
                    # --- Loop 3: Save ALL quadrants ---
                    for name_suffix, box in quad_boxes.items():
                        quadrant_img = img.crop(box)
                        
                        print(f"    -> Saving {name_suffix}")
                        new_filename = f"{base_filename}{name_suffix}.tif"
                        out_path = os.path.join(class_source_dir, new_filename)
                        quadrant_img.save(out_path, "TIFF")
                        total_quads_kept += 1
                        
                # 5. After successfully processing all 4 quads, delete the original
                print(f"  Cleaning up original file: {os.path.basename(tif_path)}")
                os.remove(tif_path)
                            
            except Exception as e:
                print(f"    Error processing {tif_path}: {e}")

    print(f"\n✅ S-Grade Quadrant Splitting complete.")
    print(f"   Split {total_files_split} parent files into {total_quads_kept} new quadrants.")
    
    if total_files_split > 0:
        print(f"\n🔥 ACTION REQUIRED: You can now manually delete any 'boring' quadrants.")
        print(f"   When ready, run this *same script again* to split them further.")
    else:
        print(f"\n🔥 ACTION REQUIRED: No more .tif files found to split.")
        print(f"   You are now ready to run 'preprocess.py' (the 'Tile Factory')!")

if __name__ == "__main__":
    main()