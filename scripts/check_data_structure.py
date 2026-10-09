from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root (the folder that contains data/ and outputs/)

DATA_DIR = ROOT / "data"  # e.g. r"C:\Users\User\skin-xai\DATA"
SPLITS = ["train", "val", "test"]  # if you still have "testing", add it here

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def count_images(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)

def main():
    print("Checking dataset structure...")
    print(f"DATA_DIR: {DATA_DIR}")
    print("-" * 60)

    for split in SPLITS:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            print(f"ERROR: Split folder not found: {split_dir}")
            continue

        class_folders = [d for d in split_dir.iterdir() if d.is_dir()]
        if not class_folders:
            print(f"ERROR: No class folders found inside: {split_dir}")
            continue

        print(f"[{split}] found {len(class_folders)} class folders:")
        for c in sorted(class_folders, key=lambda x: x.name.lower()):
            n = count_images(c)
            print(f"  - {c.name}: {n} images")

        print("-" * 60)

    print("DONE.")

if __name__ == "__main__":
    main()
