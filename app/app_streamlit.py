from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image

from torchvision import transforms
from torchvision.models import resnet18

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

ROOT = Path(__file__).resolve().parents[1]  # project root (the folder that contains data/ and outputs/)


# -------------------------
# Paths
# -------------------------
CKPT_PATH = ROOT / "outputs" / "best_resnet18.pth"
IMG_SIZE = 224


def build_model(num_classes: int):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(ckpt_path: Path, device: str):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    class_names = ckpt.get("class_names", None)
    if class_names is None:
        class_names = ckpt.get("classes", None)
    if class_names is None:
        raise KeyError("Checkpoint missing class names. Expected key: 'class_names' or 'classes'.")

    state = ckpt.get("model_state", None)
    if state is None:
        state = ckpt.get("model", None)
    if state is None:
        raise KeyError("Checkpoint missing model weights. Expected key: 'model_state' or 'model'.")

    model = build_model(num_classes=len(class_names))
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, class_names


def preprocess_for_model(img_rgb: np.ndarray):
    # img_rgb: uint8 RGB
    img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img_float = img_rgb.astype(np.float32) / 255.0

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
    tensor = tf(img_float).unsqueeze(0)  # (1,3,H,W)
    return img_float, tensor


def make_grayscale_background(rgb_float: np.ndarray):
    # rgb_float in [0,1]
    gray = cv2.cvtColor((rgb_float * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    return gray_rgb


@st.cache_resource
def get_model_and_cam():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on device: {device}")

    model, class_names = load_checkpoint(CKPT_PATH, device=device)

    # Better visual clarity: use a mid-level layer
    # If you want: try layer4[-1] too
    target_layers = [model.layer3[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    return model, class_names, cam, device


def softmax_probs(logits: torch.Tensor):
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    return probs


def main():
    st.set_page_config(page_title="Skin Disease Classifier + Grad-CAM", layout="wide")
    st.title("Skin Disease Classification with Grad-CAM (Demo)")

    st.write("Upload a face/skin image. The app will predict the class and show a Grad-CAM heatmap.")

    with st.sidebar:
        st.header("Settings")
        samples_info = st.checkbox("Show top-5 probabilities", value=True)
        heat_strength = st.slider("Heatmap strength (image weight)", 0.0, 1.0, 0.50, 0.05)
        st.caption("Lower image weight -> heatmap more visible. Higher -> original image more visible.")

    try:
        model, class_names, cam, device = get_model_and_cam()
    except Exception as e:
        st.error(f"Failed to load model/checkpoint: {e}")
        st.stop()

    uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if uploaded is None:
        st.info("Please upload an image to start.")
        return

    # Read image
    pil_img = Image.open(uploaded).convert("RGB")
    img_rgb = np.array(pil_img)  # RGB uint8

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original image")
        st.image(pil_img, use_container_width=True)

    # Inference
    rgb_float, x = preprocess_for_model(img_rgb)
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        pred_idx = int(torch.argmax(logits, dim=1).item())
        pred_name = class_names[pred_idx]
        probs = softmax_probs(logits)

    # Grad-CAM
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=x, targets=targets)[0]  # HxW

    # Improve readability: grayscale background + colored heatmap
    gray_bg = make_grayscale_background(rgb_float)
    cam_rgb = show_cam_on_image(gray_bg, grayscale_cam, use_rgb=True, image_weight=float(heat_strength))

    with col2:
        st.subheader("Prediction")
        st.markdown(f"**Predicted class:** `{pred_name}`")
        st.markdown(f"**Confidence:** `{probs[pred_idx]:.3f}`")

        if samples_info:
            st.write("Top probabilities:")
            topk = np.argsort(-probs)[:min(5, len(probs))]
            for i in topk:
                st.write(f"- {class_names[i]}: {probs[i]:.3f}")

        st.subheader("Grad-CAM (overlay on grayscale)")
        st.image(cam_rgb, use_container_width=True)

    st.divider()
    st.caption("Note: Grad-CAM highlights regions that influenced the model decision; it does not provide exact lesion boundaries.")


if __name__ == "__main__":
    main()
