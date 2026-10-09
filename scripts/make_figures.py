"""Build the figures used in the README and in docs/ from files that already exist in the project.

- docs/training_curves.png     : loss and accuracy per epoch, read from outputs/training_log.csv
- docs/gradcam_examples.png    : a grid of Grad-CAM images from outputs/gradcam
- docs/dataset_samples.png     : example images of every class (data/train)
- docs/error_gallery.png       : the test images the model got wrong, from outputs/test_predictions.csv
- docs/gradcam_layer_comparison.png: Grad-CAM on layer4 (first attempt) versus layer3 (final choice)
- docs/confidence.png          : not produced here (see src/evaluate.py)

Usage:
    python scripts/make_figures.py
"""
import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)
CLASSES = ["Acne", "Actinic Keratosis", "Basal Cell Carcinoma", "Eczema", "Rosacea"]
IMG_EXT = {".jpg", ".jpeg", ".png"}


def read_epochs(log_path):
    rows = []
    with open(log_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and row[0].isdigit():
                rows.append([float(x) for x in row[:6]])
    return rows


def training_curves():
    rows = read_epochs(ROOT / "outputs" / "training_log.csv")
    epoch = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epoch, [r[1] for r in rows], marker="o", label="train")
    axes[0].plot(epoch, [r[3] for r in rows], marker="o", label="validation")
    axes[0].set(title="Loss", xlabel="epoch"); axes[0].legend()
    axes[1].plot(epoch, [r[2] for r in rows], marker="o", label="train")
    axes[1].plot(epoch, [r[4] for r in rows], marker="o", label="validation")
    axes[1].plot(epoch, [r[5] for r in rows], marker="o", ls="--", label="validation macro-F1")
    axes[1].set(title="Accuracy", xlabel="epoch", ylim=(0.4, 1.02)); axes[1].legend()
    fig.tight_layout()
    fig.savefig(DOCS / "training_curves.png", dpi=130)
    plt.close(fig)


def gradcam_grid():
    files = sorted((ROOT / "outputs" / "gradcam").glob("*.jpg"))
    fig, axes = plt.subplots(3, 5, figsize=(16, 10))
    for ax, f in zip(axes.flat, files):
        true, pred = f.stem.split("__")[0][5:], f.stem.split("__")[1][5:]
        ax.imshow(Image.open(f)); ax.axis("off")
        ax.set_title(f"true: {true}\npredicted: {pred}", fontsize=8, color="green" if true == pred else "red")
    for ax in list(axes.flat)[len(files):]:
        ax.axis("off")
    fig.suptitle("Grad-CAM on test images (green = correct prediction, red = wrong)", fontsize=12)
    fig.tight_layout()
    fig.savefig(DOCS / "gradcam_examples.png", dpi=110)
    plt.close(fig)


def dataset_samples(per_class=5, seed=7):
    rng = random.Random(seed)
    fig, axes = plt.subplots(len(CLASSES), per_class, figsize=(2.4 * per_class, 2.5 * len(CLASSES)))
    for r, cls in enumerate(CLASSES):
        files = sorted(p for p in (ROOT / "data" / "train" / cls).iterdir() if p.suffix.lower() in IMG_EXT)
        for c, f in enumerate(rng.sample(files, per_class)):
            ax = axes[r][c]
            ax.imshow(Image.open(f).convert("RGB")); ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(cls, fontsize=9)
    fig.suptitle("Example training images of each class", fontsize=12)
    fig.tight_layout()
    fig.savefig(DOCS / "dataset_samples.png", dpi=100)
    plt.close(fig)


def error_gallery(max_images=40):
    path = ROOT / "outputs" / "test_predictions.csv"
    if not path.exists():
        print("outputs/test_predictions.csv not found; run python src/evaluate.py first")
        return
    with open(path, newline="", encoding="utf-8") as f:
        wrong = [r for r in csv.DictReader(f) if r["correct"] == "0"][:max_images]
    cols = 8
    rows = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.6 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, wrong):
        img = next((p for p in (ROOT / "data" / "test" / r["true"]).iterdir() if p.name == r["file"]), None)
        if img:
            ax.imshow(Image.open(img).convert("RGB"))
        short = lambda s: "".join(w[0] for w in s.split()) if " " in s else s
        ax.set_title(f"true: {short(r['true'])}\npred: {short(r['predicted'])}", fontsize=7.5, color="red")
    fig.suptitle(f"Test images the model got wrong ({len(wrong)} of 185)   AK = Actinic Keratosis, BCC = Basal Cell Carcinoma", fontsize=11)
    fig.tight_layout()
    fig.savefig(DOCS / "error_gallery.png", dpi=95)
    plt.close(fig)


def layer_comparison():
    """Same test images explained with layer4[-1] (outputs/gradcam_layer4) and layer3[-1] (outputs/gradcam)."""
    picks = ["acne-cystic-13", "actinic-keratosis-lesion-31", "actinic-keratosis-face-9",
             "basal-cell-carcinoma-face-4", "eczema23", "rosacea-44"]
    fig, axes = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 5.8))
    for c, key in enumerate(picks):
        for r, folder in enumerate(("gradcam_layer4", "gradcam")):
            f = next((ROOT / "outputs" / folder).glob(f"*{key}.jpg"))
            true, pred = f.stem.split("__")[0][5:], f.stem.split("__")[1][5:]
            axes[r][c].imshow(Image.open(f)); axes[r][c].axis("off")
            if r == 0:
                axes[r][c].set_title(f"true: {true}\npred: {pred}", fontsize=7, color="green" if true == pred else "red")
    axes[0][0].text(-0.02, 0.5, "layer4 (7x7)", transform=axes[0][0].transAxes, rotation=90, va="center", ha="right", fontsize=10)
    axes[1][0].text(-0.02, 0.5, "layer3 (14x14)", transform=axes[1][0].transAxes, rotation=90, va="center", ha="right", fontsize=10)
    fig.suptitle("Grad-CAM: last block of layer4 (first attempt, top) vs layer3 (final choice, bottom)", fontsize=11)
    fig.tight_layout()
    fig.savefig(DOCS / "gradcam_layer_comparison.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    training_curves()
    gradcam_grid()
    dataset_samples()
    error_gallery()
    layer_comparison()
    print("figures written to docs/")
