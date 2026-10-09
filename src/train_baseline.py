import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

ROOT = Path(__file__).resolve().parents[1]  # project root (the folder that contains data/ and outputs/)


def build_loaders(data_dir: Path, img_size=224, batch_size=16, num_workers=2):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=7),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    train_ds = datasets.ImageFolder(data_dir / "train", train_tf)
    val_ds   = datasets.ImageFolder(data_dir / "val", eval_tf)
    test_ds  = datasets.ImageFolder(data_dir / "test", eval_tf)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds.classes


def build_model(num_classes):
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds, all_targets = [], []
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    return (
        total_loss / len(loader.dataset),
        accuracy_score(all_targets, all_preds),
        f1_score(all_targets, all_preds, average="macro"),
        all_targets,
        all_preds
    )


def main():
    DATA_DIR = ROOT / "data"
    OUTPUT_DIR = ROOT / "outputs"
    OUTPUT_DIR.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, classes = build_loaders(DATA_DIR)
    model = build_model(len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    log_path = OUTPUT_DIR / "training_log.csv"
    best_path = OUTPUT_DIR / "best_resnet18.pth"

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# TRAINING LOG"])
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_macro_f1"])

    best_f1 = -1

    for epoch in range(1, 16):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, tr_loss, tr_acc, val_loss, val_acc, val_f1])

        print(f"Epoch {epoch} | train_acc={tr_acc:.3f} val_acc={val_acc:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"model": model.state_dict(), "classes": classes}, best_path)

    # ---- TEST PHASE ----
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, device)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)

        w.writerow([])
        w.writerow(["# TEST SUMMARY"])
        w.writerow(["metric", "value"])
        w.writerow(["test_loss", test_loss])
        w.writerow(["test_accuracy", test_acc])
        w.writerow(["test_macro_f1", test_f1])

        w.writerow([])
        w.writerow(["# CLASSIFICATION REPORT"])
        w.writerow(["class", "precision", "recall", "f1_score", "support"])
        for cls in classes:
            r = report[cls]
            w.writerow([cls, r["precision"], r["recall"], r["f1-score"], r["support"]])

        w.writerow([])
        w.writerow(["# CONFUSION MATRIX"])
        w.writerow(["true\\pred"] + classes)
        for i, row in enumerate(cm):
            w.writerow([classes[i]] + row.tolist())

    print("SUCCESS: All training and test results saved to training_log.csv")


if __name__ == "__main__":
    main()
