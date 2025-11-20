import os
import random
import json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --- CONFIGURATION ---
# Defines the structure of your "Balanced" subset
CLASS_SAMPLE_MAP = {
    "urban": 6000,
    "Agricultural": 3500,
    "mountain": 2000,
    "Forest": 3500,
    "Grassland": 3000,
    "Beach": 2500,
    "port": 4000,
    "key west": 2000,
    "Desert": 1000,
    "Airport": 3000,
    "salt_utha": 1000, 
    "new york": 6000,
    "revier_delta": 1500,
    "university": 3500,
    "platue": 1500,
    "moutntainsnow":3000,
    "lake": 2000,
    "oill": 1000,
    "island": 1000,
    "stadium": 2500,
    "solar park": 800,
}

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128), Image.BICUBIC),
    transforms.ToTensor(),
])

# --- HELPER CLASSES ---

class _ListDataset(Dataset):
    def __init__(self, files, transform, overlap=True, overlap_px=16, image_size=128):
        self.files = files
        self.transform = transform
        self.overlap = overlap
        self.overlap_px = overlap_px
        self.image_size = image_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.files[idx]).convert("RGB")
            if self.overlap:
                arr = np.array(img)
                pad = self.overlap_px
                padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
                ox = np.random.randint(0, 2 * pad + 1)
                oy = np.random.randint(0, 2 * pad + 1)
                crop = padded[oy:oy + 128, ox:ox + 128]
                img = Image.fromarray(crop)
            
            img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
            return self.transform(img), 0
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            return torch.zeros((3, self.image_size, self.image_size)), 0

class TileFolderDataset(Dataset):
    def __init__(self, folder_path):
        self.files = []
        if os.path.exists(folder_path):
            for root, _, files in os.walk(folder_path):
                for f in files:
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                        self.files.append(os.path.join(root, f))

# --- MAIN FUNCTIONS ---

def _find_class_folders(root_dir):
    mapping = {}
    if not os.path.exists(root_dir):
        return mapping
    for entry in os.scandir(root_dir):
        if entry.is_dir():
            mapping[entry.name.strip().lower()] = entry.path
    return mapping

def _build_balanced_index(root_dir, class_map):
    # <--- NEW FEATURE: THE STICKY LIST --->
    split_file = "data/train_split_frozen.json"
    
    # 1. If list exists, LOAD IT. Do not re-shuffle.
    if os.path.exists(split_file):
        print(f"[dataset] 🔒 Loading FROZEN split from {split_file}")
        with open(split_file, "r") as f:
            selected = json.load(f)
        print(f"[dataset] Loaded {len(selected)} images.")
        return selected

    # 2. If no list, GENERATE IT (Once).
    print(f"[dataset] 🎲 Generating NEW random split...")
    folder_map = _find_class_folders(root_dir)
    selected = []
    stats = []
    
    for cls_name, count in class_map.items():
        key = cls_name.lower()
        if key not in folder_map:
            stats.append((cls_name, 0, 0))
            continue

        ds = TileFolderDataset(folder_map[key])
        n_avail = len(ds.files)
        n_pick = min(n_avail, count)
        
        if n_pick == 0:
            stats.append((cls_name, 0, 0))
            continue

        idxs = list(range(n_avail))
        random.shuffle(idxs)
        files = [ds.files[i] for i in idxs[:n_pick]]
        selected.extend(files)
        stats.append((cls_name, n_pick, n_avail))

    random.shuffle(selected)

    # 3. Save it so we never change it again.
    os.makedirs("data", exist_ok=True)
    with open(split_file, "w") as f:
        json.dump(selected, f)
    print(f"[dataset] 💾 Saved frozen split to {split_file}")

    # Print Stats
    print("\n" + "="*65)
    print(f"| {'CLASS NAME':<25} | {'SELECTED':<10} | {'AVAILABLE':<10} |")
    print("|" + "-"*63 + "|")
    for cls, sel, avail in stats:
        print(f"| {cls:<25} | {sel:<10} | {avail:<10} |")
    print("|" + "-"*63 + "|")
    print(f"| {'TOTAL DATASET':<25} | {len(selected):<10} | {'-':<10} |")
    print("="*65 + "\n")

    return selected

def get_balanced_dataloader(data_dir, image_size=128, batch_size=8, overlap=True, overlap_px=16, num_workers=4):
    files = _build_balanced_index(data_dir, CLASS_SAMPLE_MAP)
    if not files or len(files) == 0:
        print("[dataset] CRITICAL: No images found. Check DATA_DIR path.")
        return None

    ds = _ListDataset(files, TRAIN_TRANSFORM, overlap, overlap_px, image_size)
    
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=True, # Shuffle BATCHES, not the dataset selection
        num_workers=num_workers, 
        pin_memory=True
    ) 