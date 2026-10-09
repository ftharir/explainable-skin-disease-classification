# Dataset

Five classes of facial skin conditions, organised for `torchvision.datasets.ImageFolder`:

```
data/
├── train/{Acne, Actinic Keratosis, Basal Cell Carcinoma, Eczema, Rosacea}/   75 images per class
├── val/  {same five classes}/                                                13 images per class
└── test/ {same five classes}/                                                37 images per class
```

| Split | Images | Used for |
|---|---|---|
| train | 375 | fitting the model |
| val | 65 | choosing the best epoch (highest macro-F1) |
| test | 185 | final evaluation only |
| **total** | **625** | |

## Source

Kaggle: [`amellia/face-skin-disease`](https://www.kaggle.com/datasets/amellia/face-skin-disease). The original dataset comes with a `train` and a `testing` folder; the validation split was cut out of `train` with `scripts/make_val_and_fix_folders.py` (15% per class, seed 42) and `testing` was renamed to `test`. To download the data again, run `python scripts/download_dataset.py` (needs a Kaggle account).

The images appear to originate from DermNet and carry a "©Dermnet.com" watermark. They belong to their original owners and are included here only so that the results of this educational project can be reproduced; please check the terms of the source before reusing them.

## What the classes look like

| Class | Typical appearance |
|---|---|
| Acne | inflamed pimples, blackheads and whiteheads, local redness |
| Actinic Keratosis | rough, dry, scaly patches on sun-exposed skin (a pre-cancerous lesion) |
| Basal Cell Carcinoma | pearly or waxy bump, sometimes with visible small vessels (a skin cancer) |
| Eczema | dry, flaky, red and itchy patches |
| Rosacea | persistent diffuse redness of the central face, sometimes with small bumps |

![Example images](../docs/dataset_samples.png)

## Properties and caveats

- Perfectly balanced classes (125 images each) in a 75 / 13 / 37 split.
- Images are JPEG and PNG with varying size and aspect ratio (resized to 224 × 224 for the model). A few files are `.jpeg` or `.png`; all are used.
- Some images show the whole face, others a close-up of a small region, and some have black bars over the eyes.
- 7 groups of identical images exist, 2 of which are spread across different splits (a small source of leakage). The split is by image, not by patient.
- No metadata (age, sex, skin tone) is available.
- Use `python scripts/check_data_structure.py` to print the number of images per split and class.
