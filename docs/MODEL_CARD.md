# Model card: facial skin condition classifier

## Model details

| | |
|---|---|
| **Task** | Image classification into 5 facial skin conditions: Acne, Actinic Keratosis, Basal Cell Carcinoma, Eczema, Rosacea |
| **Architecture** | ResNet18 (torchvision), ImageNet-pretrained, final layer replaced by a 5-way linear layer, all layers fine-tuned |
| **Input** | RGB image resized to 224 × 224, ImageNet mean / std normalisation |
| **Output** | Class probabilities (softmax) and a Grad-CAM heatmap for the predicted class |
| **Checkpoint** | `outputs/best_resnet18.pth` (about 45 MB; contains the weights and the class names) |
| **Training** | Adam, learning rate 1e-4, batch size 16, 15 epochs; checkpoint with the best validation macro-F1 kept |
| **Augmentation** | Horizontal flip, rotation ±7°, colour jitter (training only) |
| **Framework** | PyTorch, torchvision, `pytorch-grad-cam` |
| **Developed** | December 2025 – mid-2026, as a bachelor's thesis project |

## Intended use

- Education and research on medical image classification and explainable AI.
- A demonstration of how a Grad-CAM explanation can be shown next to a prediction.

## Out-of-scope use

- **Any diagnostic or treatment decision.** The model was not clinically validated and has not been reviewed by a dermatologist.
- Skin conditions other than the five classes: the model always answers with one of them, even for a healthy face or for an unrelated image.
- Images that are not close-up photos of facial skin.

## Training and evaluation data

Kaggle dataset `amellia/face-skin-disease`: 625 photos, 125 per class, split into 375 train / 65 validation / 185 test images (75 / 13 / 37 per class). The images appear to come from DermNet and carry its watermark. See [`data/README.md`](../data/README.md).

## Metrics (held-out test set, 185 images)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Acne | 0.84 | 0.86 | 0.85 |
| Actinic Keratosis | 0.88 | 0.62 | 0.73 |
| Basal Cell Carcinoma | 0.71 | 0.97 | 0.82 |
| Eczema | 0.83 | 0.78 | 0.81 |
| Rosacea | 0.80 | 0.76 | 0.78 |
| **Overall** | accuracy **0.800** | macro-F1 **0.797** | |

Reproduce with `python src/evaluate.py` (writes `outputs/test_metrics.json`).

## Known failure modes

37 of the 185 test images are misclassified (see [`error_gallery.png`](error_gallery.png)). The most frequent confusions:

| True → predicted | Count |
|---|---|
| Actinic Keratosis → Basal Cell Carcinoma | 7 |
| Actinic Keratosis → Rosacea | 5 |
| Rosacea → Basal Cell Carcinoma | 4 |
| Rosacea → Acne | 4 |
| Acne → Eczema | 3 |
| Eczema → Actinic Keratosis | 3 |

- **Actinic Keratosis is under-detected** (recall 62%) and is often read as Basal Cell Carcinoma or Rosacea. Basal Cell Carcinoma, in turn, receives many false alarms (precision 71%), which means the model leans towards the more serious diagnosis when unsure.
- Heatmaps in wrong predictions are often diffuse or fall on non-lesion areas such as eyebrows, jawline or background (see [`gradcam_examples.png`](gradcam_examples.png)).

## Ethical considerations and limitations

- **Small, web-sourced data.** 375 training images are far too few to cover the variety of real patients, lighting, cameras and disease stages. The model overfits the training set (training accuracy reaches 100%).
- **Skin tone and demographic coverage are unknown.** The dataset has no metadata, so performance for different skin tones, ages or sexes has not been measured and may differ.
- **Watermarks and image style.** All images share the same source style and watermark, so the model may partly rely on image style rather than on the condition itself.
- **Possible leakage.** A few duplicated images exist in the dataset (7 groups, 2 of them across splits) and the split is by image, not by patient.
- **No probability calibration.** The confidence value shown in the demo should not be read as the real probability of a diagnosis.
- **Explanations are approximate.** Grad-CAM shows where the network looked, not the lesion boundary, and a plausible-looking heatmap does not prove a prediction is correct.
- Results come from a single training run without a fixed seed or cross-validation.
