# AN2DL Challenges: Politecnico di Milano, A.Y. 2025/2026

Solutions to the two Kaggle challenges of the course **Artificial Neural Networks and Deep Learning (AN2DL)**.

**Team NoMoreTokensInLA**: Lorenzo Esposito, Alessandro Frisone, Francesco Grassi, Giorgio Venezia

All notebooks were developed and run on Google Colab, with data and checkpoints stored on a shared Google Drive. They are kept as they ran, including outputs.

---

## First Challenge: pain classification from time series

Multivariate time-series classification of each subject into `no_pain`, `low_pain` or `high_pain`. The classes are strongly imbalanced (the final submission has about 78% `no_pain`).

**Final model**: a bidirectional LSTM trained on sliding windows.

| Hyperparameter | Value |
|---|---|
| Window size / stride | 32 / 4 |
| Hidden size × layers | 128 × 2 |
| Dropout | 0.3 |
| Loss | Focal Loss (γ = 1.75) |
| Learning rate | 1e-3 |
| L2 regularisation | 1e-4 |
| Batch size | 64 |
| Early stopping patience | 20 |

| Notebook | Validation F1 |
|---|---|
| [`notebooks/f10.9761`](First_Challenge/notebooks/f10.9761) (**final**) | **0.9761** |
| [`notebooks/f10.97`](First_Challenge/notebooks/f10.97) | 0.9706 |

Each folder contains the notebook, the trained weights (`.pt`) and the submission CSV it produced. The report is linked in [`report/link_to_report`](First_Challenge/report/link_to_report).

---

## Second Challenge: breast cancer molecular subtype classification

Classification of histological images into 4 molecular subtypes: Luminal A, Luminal B, HER2(+) and Triple negative. The data is 691 training images and 477 test images, each with a tissue mask. Full details are in the [report](Second_Challenge/AN2DL_Report.pdf).

**Pipeline**
1. **Data cleaning**: images whose green channel dominates the blue one (non-tissue outliers and green-stain artifacts) are removed, going from 691 to 581 samples.
2. **Masking and patching**: the background is removed with the mask, and one representative patch is kept per image, chosen to maximise the tissue ratio and resized to 256 × 256. A multi-tile variant was also tried but generalised worse.
3. **Normalisation**: Macenko stain normalisation (`torchstain`) followed by ImageNet normalisation. Augmentation is geometric only (flips, rotations, translations), with no colour jitter.
4. **Model**: ResNet-50 pre-trained on ImageNet, with a single linear head (2048 → 4).
5. **Training**: phase 1 trains the head with the backbone frozen, phase 2 fine-tunes the last residual blocks. The optimiser is AdamW (lr 1e-3, weight decay 1e-5) with ReduceLROnPlateau (factor 0.5).

**Results** (from the report)

| Model | Train F1 | Val F1 | Kaggle |
|---|---|---|---|
| EfficientNetV2 | 0.32 | 0.33 | 0.28 |
| ResNet18 | 0.32 | 0.27 | 0.29 |
| ResNet50 (unbalanced) | 0.52 | 0.40 | 0.36 |
| **ResNet50 (balanced)** | **0.47** | **0.41** | **0.40** |
| ResNet152 (balanced) | 0.92 | 0.63 | 0.24 |

> **Note on the notebook.** [`notebooks/Challenge_Tiled_Last.ipynb`](Second_Challenge/notebooks/Challenge_Tiled_Last.ipynb) is the last complete run before the deadline. It contains the full pipeline above and produced [`submissions/submission_eff.csv`](Second_Challenge/submissions/submission_eff.csv). The exact run behind the Kaggle score of 0.40 was not saved on the shared Drive, so the outputs stored in this notebook differ from the report: best validation accuracy is 0.38, and this run fine-tunes three residual blocks without class balancing.

---

## Repository structure

```
First_Challenge/
├── notebooks/
│   ├── f10.9761/      # final model: notebook, weights, submission
│   └── f10.97/        # variant
└── report/link_to_report
Second_Challenge/
├── AN2DL_Report.pdf
├── notebooks/Challenge_Tiled_Last.ipynb
└── submissions/submission_eff.csv
```

## Data and weights

The competition datasets and most trained checkpoints (several GB) are **not** included in this repository. To re-run the notebooks, download the data from the course Kaggle competitions and place it under `dataset/` as expected by the data-loading cells. You will also need to adapt the Google Drive paths in the first cells.
