# Note on the numbers in the thesis

The thesis was written after the experiments. Its headline results agree with what this repository reproduces (**accuracy 80%, macro-F1 ≈ 0.80**), but several detailed numbers in Chapter 4 and some implementation details differ from the files in this repository. **The numbers in the repository [README](../../README.md) are the authoritative ones**: they can be reproduced with `python src/evaluate.py` from the saved checkpoint, and they match the training log (`outputs/training_log.csv`), which is also shown as a screenshot in the thesis (Figure 3.3 / 4.1).

The thesis documents are published as written; this note lists the differences so that readers can interpret them correctly.

## Results

| Item | Thesis text | This repository (reproducible) |
|---|---|---|
| Test accuracy | 80% | 80.0% (148 / 185) |
| Macro-F1 | 0.80 | 0.797 |
| Best epoch (by validation macro-F1) | about epoch 4 | epoch 5 (val macro-F1 0.816) |
| Epoch table (Table 4.1) | e.g. epoch 1: train loss 1.2413, val acc 69.23%; epoch 15: val acc 64.62%, val macro-F1 0.648 | epoch 1: train loss 1.2120, val acc 70.77%; epoch 15: val acc 75.38%, val macro-F1 0.747 |

### Classification report on the test set

| Class | Thesis (precision / recall) | Repository (precision / recall) |
|---|---|---|
| Acne | 0.810 / 0.919 | 0.842 / 0.865 |
| Actinic Keratosis | 0.957 / 0.595 | 0.885 / 0.622 |
| Basal Cell Carcinoma | 0.739 / 0.919 | 0.706 / 0.973 |
| Eczema | 0.769 / 0.811 | 0.829 / 0.784 |
| Rosacea | 0.800 / 0.757 | 0.800 / 0.757 |

### Confusion matrix

The matrix in the thesis text (Table 4.3) has 150 correct predictions out of 185, i.e. 81.1%, which is not the 80% reported in the same chapter. The matrix of the repository (and the spreadsheet screenshot in the thesis, Figure 4.1) has 148 correct predictions, which is 80.0%:

```
true \ predicted   Acne  AK  BCC  Eczema  Rosacea
Acne                32    0    2     3       0
Actinic Keratosis    1   23    7     1       5
Basal Cell Carc.     0    0   36     1       0
Eczema               1    3    2    29       2
Rosacea              4    0    4     1      28
```

The error table in the thesis (Table 4.4) follows its own matrix; in the repository the Actinic Keratosis → Rosacea count is 5 (not 4) and there are no Rosacea → Actinic Keratosis errors (Rosacea errors go to Acne and Basal Cell Carcinoma, 4 each). The qualitative conclusions (Actinic Keratosis is the hardest class and is mostly confused with Basal Cell Carcinoma; Acne and Basal Cell Carcinoma are the easiest) hold in both.

## Implementation details

| Item | Thesis text | Code in this repository |
|---|---|---|
| Rotation augmentation | ±10° | ±7° |
| Brightness augmentation | ±0.2 | brightness 0.15, contrast 0.15, saturation 0.10 |
| Grad-CAM library | `torch-cam` | `pytorch-grad-cam` |
| Grad-CAM layer | "intermediate layer (layer 4), 14×14 instead of 7×7" | `layer3[-1]` (its output is 14×14 for a 224×224 input; `layer4` would give 7×7) |

## Other editorial notes

- The Persian heading of Chapter 3 reads "Discussion and conclusion", but the chapter describes the methodology; the English edition titles it "Methodology".
- Esteva et al. (2017) is cited in the text but missing from the reference list, in both editions.
