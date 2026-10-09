import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root (the folder that contains data/ and outputs/)

# Path to your DATA folder (the one that contains 'train' and 'testing')
DATA_DIR = ROOT / "data"  # e.g. r"C:\Users\User\skin-xai\DATA"

TRAIN_DIR = DATA_DIR / "train"
TESTING_DIR = DATA_DIR / "testing"
TEST_DIR = DATA_DIR / "test"
VAL_DIR = DATA_DIR / "val"

# Set this to 0.15 to create 15% validation split from train
VAL_RATIO = 0.15

# If True: move files from train -> val (recommended to avoid duplicates)
# If False: copy files from train -> val (keeps train unchanged)
MOVE_TO_VAL = True

SEED = 42
random.seed(SEED)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(folder: Path):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

def ensure_exists(p: Path, name: str):
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {p}")

def main():
    print("Starting folder checks...")
    print(f"DATA_DIR: {DATA_DIR}")

    ensure_exists(TRAIN_DIR, "train folder")

    if not TEST_DIR.exists():
        if TESTING_DIR.exists():
            print("Found 'testing' folder. Renaming it to 'test'...")
            TESTING_DIR.rename(TEST_DIR)
            print("OK: 'testing' -> 'test'")
        else:
            print("WARNING: Neither 'test' nor 'testing' folder exists. Test split will be missing.")
    else:
        print("OK: 'test' folder already exists.")

    # Create val folder
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    # Detect classes from train directory
    classes = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]
    if not classes:
        print("ERROR: No class folders found inside 'train'.")
        return

    print("Detected class folders in train:")
    for c in classes:
        print(f"  - {c.name}")

    print("\nCreating validation split from train...")
    print(f"VAL_RATIO: {VAL_RATIO}")
    print(f"MOVE_TO_VAL: {MOVE_TO_VAL}")
    print(f"SEED: {SEED}")
    print("-" * 60)

    total_moved = 0
    for class_dir in classes:
        class_name = class_dir.name
        imgs = list_images(class_dir)

        if len(imgs) < 5:
            print(f"WARNING: Very few images in class '{class_name}': {len(imgs)}")
            continue

        random.shuffle(imgs)
        n_val = max(1, int(len(imgs) * VAL_RATIO))

        val_imgs = imgs[:n_val]
        out_class_dir = VAL_DIR / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        for img in val_imgs:
            dst = out_class_dir / img.name
            if MOVE_TO_VAL:
                shutil.move(str(img), str(dst))
            else:
                shutil.copy2(str(img), str(dst))

        total_moved += len(val_imgs)
        print(f"OK: {class_name} | moved_to_val={len(val_imgs)} | remaining_in_train={len(imgs)-len(val_imgs)}")

    print("-" * 60)
    print(f"DONE: Total files moved/copied to val: {total_moved}")
    print("SUCCESS: You now have train/val/test under DATA_DIR.")

if __name__ == "__main__":
    main()
