"""Evaluate the saved checkpoint on the validation and test sets.

Writes (in outputs/): test_metrics.json, test_predictions.csv, and docs/confusion_matrix.png.

Usage (from anywhere):
    python src/evaluate.py
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train_baseline import build_model, evaluate

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CKPT_PATH = ROOT / "outputs" / "best_resnet18.pth"
IMG_SIZE = 224


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CKPT_PATH, map_location=device)
    classes = ckpt["classes"]
    model = build_model(len(classes)).to(device)
    model.load_state_dict(ckpt["model"])

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    metrics = {}
    for split in ("val", "test"):
        ds = datasets.ImageFolder(DATA_DIR / split, eval_tf)
        assert ds.classes == classes
        loss, acc, f1, y_true, y_pred = evaluate(model, DataLoader(ds, batch_size=16), device)
        metrics[split] = {"images": len(ds), "loss": round(loss, 4),
                          "accuracy": round(acc, 4), "macro_f1": round(f1, 4)}
        if split == "test":
            test_ds, test_true, test_pred = ds, y_true, y_pred

    report = classification_report(test_true, test_pred, target_names=classes, output_dict=True)
    metrics["test_per_class"] = {
        c: {k: round(report[c][k], 3) for k in ("precision", "recall", "f1-score")} | {"support": int(report[c]["support"])}
        for c in classes
    }
    (ROOT / "outputs" / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with open(ROOT / "outputs" / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "true", "predicted", "correct"])
        for (path, _), t, p in zip(test_ds.samples, test_true, test_pred):
            w.writerow([Path(path).name, classes[t], classes[p], int(t == p)])

    cm = confusion_matrix(test_true, test_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, fmt, title in (
        (axes[0], cm, "d", "Confusion matrix (counts)"),
        (axes[1], cm / cm.sum(axis=1, keepdims=True), ".2f", "Confusion matrix (row-normalised = recall)"),
    ):
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > data.max() / 2 else "black")
    fig.tight_layout()
    (ROOT / "docs").mkdir(exist_ok=True)
    fig.savefig(ROOT / "docs" / "confusion_matrix.png", dpi=130)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
