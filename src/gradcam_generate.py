import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

ROOT = Path(__file__).resolve().parents[1]  # project root (the folder that contains data/ and outputs/)


DATA_DIR = ROOT / "data"
CKPT_PATH = ROOT / "outputs" / "best_resnet18.pth"
OUT_DIR = ROOT / "outputs" / "gradcam"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# How many images per class to visualize
SAMPLES_PER_CLASS = 3

IMG_SIZE = 224
SEED = 42
random.seed(SEED)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_model(num_classes: int):
    model = resnet18(weights=None)  # weights come from your checkpoint
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def list_images(folder: Path):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]


def preprocess_image(img_bgr: np.ndarray):
    # BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # float in [0,1] for overlay
    img_float = img_rgb.astype(np.float32) / 255.0

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
    tensor = tf(img_float).unsqueeze(0)  # (1,3,H,W)
    return img_float, tensor


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Starting Grad-CAM generation...")
    print(f"Device: {device}")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Output folder: {OUT_DIR}")

    if not CKPT_PATH.exists():
        print("ERROR: Checkpoint file not found.")
        return

    ckpt = torch.load(CKPT_PATH, map_location=device)
    class_names = ckpt.get("class_names", None)
    if class_names is None:
        # fallback if you saved under another key name
        class_names = ckpt.get("classes", None)

    if class_names is None:
        print("ERROR: Could not find class names inside the checkpoint.")
        print("Expected key: 'class_names' or 'classes'")
        return

    num_classes = len(class_names)
    print("Classes from checkpoint:", class_names)

    model = build_model(num_classes=num_classes)
    # checkpoint key could be model_state or model
    state = ckpt.get("model_state", None)
    if state is None:
        state = ckpt.get("model", None)

    if state is None:
        print("ERROR: Could not find model weights inside the checkpoint.")
        print("Expected key: 'model_state' or 'model'")
        return

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # For ResNet18, target layer usually works well as the last block in layer4
    target_layers = [model.layer3[-1]] #ChangeForTheBetterResults #[model.layer4[-1]] --> [model.layer3[-1]] 
    cam = GradCAM(model=model, target_layers=target_layers)

    test_root = DATA_DIR / "test"
    if not test_root.exists():
        print(f"ERROR: Test folder not found: {test_root}")
        return

    # class folders under DATA/test
    test_class_dirs = [d for d in test_root.iterdir() if d.is_dir()]
    if not test_class_dirs:
        print("ERROR: No class folders found under DATA/test.")
        return

    print("Detected test class folders:", [d.name for d in sorted(test_class_dirs, key=lambda x: x.name.lower())])
    print(f"Samples per class: {SAMPLES_PER_CLASS}")
    print("-" * 60)

    total_written = 0

    for class_dir in sorted(test_class_dirs, key=lambda x: x.name.lower()):
        cls_name = class_dir.name
        imgs = list_images(class_dir)

        if len(imgs) == 0:
            print(f"WARNING: No images found for class folder: {cls_name}")
            continue

        random.shuffle(imgs)
        take = imgs[:min(SAMPLES_PER_CLASS, len(imgs))]

        for img_path in take:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"WARNING: Failed to read image: {img_path}")
                continue

            rgb_float, input_tensor = preprocess_image(img_bgr)
            input_tensor = input_tensor.to(device)

            # Predict
            with torch.no_grad():
                logits = model(input_tensor)
                pred_idx = int(torch.argmax(logits, dim=1).item())
                pred_name = class_names[pred_idx]

            # Grad-CAM for predicted class
            targets = [ClassifierOutputTarget(pred_idx)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # HxW

            cam_image = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

            # Save result
            out_name = f"true_{cls_name}__pred_{pred_name}__{img_path.stem}.jpg"
            out_path = OUT_DIR / out_name

            cam_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_path), cam_bgr)

            total_written += 1
            print(f"Saved: {out_path}")

        print(f"OK: Finished class folder: {cls_name}")

    print("-" * 60)
    print(f"DONE. Total Grad-CAM images saved: {total_written}")
    print("You can open the images in the output folder to inspect the heatmaps.")


if __name__ == "__main__":
    main()
