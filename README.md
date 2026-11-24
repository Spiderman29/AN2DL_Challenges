# AN2DL Challenges: Politecnico di Milano, A.Y. 2025/2026

Solutions to the two Kaggle challenges of the course **Artificial Neural Networks and Deep Learning (AN2DL)**.

**Team**: Lorenzo Esposito, Alessandro Frisone, Francesco Grassi, Giorgio Venezia

All notebooks were developed and run on Google Colab, with data stored on Google Drive. They are kept as they were submitted, including outputs.

---

## First Challenge: pain classification from time series

Multivariate time-series classification of 661 training users (1324 in the test set), each recorded for 160 timestamps. Every user is assigned one of `no_pain`, `low_pain` or `high_pain`, and the classes are strongly imbalanced. Full details are in the [report](First_Challenge/report/AN2DL_Report.pdf).

**Pipeline**
1. **Feature engineering**: cyclical time features from the dominant period, estimated with Welch's PSD. The `n_legs`, `n_hands` and `n_eyes` columns are merged into a single binary `is_pirate` feature, and the low-variance joint is dropped.
2. **Preprocessing**: Gaussian jittering for augmentation, a StandardScaler fit on the training data only, removal of highly correlated features, and PCA keeping 95% of the variance.
3. **Windowing**: overlapping sliding windows per user.
4. **Model**: a hybrid **CNN + bidirectional LSTM**. A 1D-CNN encoder (two Conv/BatchNorm/ReLU/MaxPool blocks) feeds a BiLSTM with an attention mechanism over its outputs.
5. **Imbalance handling**: Focal Loss with inverse-frequency class weights, plus a WeightedRandomSampler.
6. **Training**: AdamW, gradient clipping, ReduceLROnPlateau on the validation F1, and early stopping.

**Results** (from the report)

| Model | Validation F1 |
|---|---|
| B-LSTM | 0.9318 |
| B-LSTM + Focal Loss | 0.9594 |
| **CNN + B-LSTM + Focal Loss** | **0.9701** |

The submitted notebook is [`notebooks/NaN_First_challenge.ipynb`](First_Challenge/notebooks/NaN_First_challenge.ipynb). The `f10.97` and `f10.9761` folders contain earlier BiLSTM experiments, each with its notebook, weights and submission.

---

## Second Challenge: breast cancer molecular subtype classification

Classification of histological images into 4 molecular subtypes: Luminal A, Luminal B, HER2(+) and Triple negative. The data is 691 training images and 477 test images, each with a tissue mask. Full details are in the [report](Second_Challenge/AN2DL_Report.pdf).

**Pipeline**
1. **Data cleaning**: images whose green channel dominates the blue one (non-tissue outliers and green-stain artifacts) are removed, going from 691 to 581 samples.
2. **Masking and patching**: the background is removed with the mask, and one representative patch is kept per image, chosen to maximise the tissue ratio and resized to 256 × 256. A multi-tile variant was also tried but generalised worse.
3. **Normalisation**: Macenko stain normalisation (`torchstain`) followed by ImageNet normalisation. Augmentation is geometric only, with no colour jitter.
4. **Model**: ResNet-50 pre-trained on ImageNet, with a single linear head (2048 → 4).
5. **Training**: phase 1 trains the head with the backbone frozen, phase 2 fine-tunes the residual blocks. The optimiser is AdamW (lr 1e-3, weight decay 1e-5) with ReduceLROnPlateau (factor 0.5).

**Results** (from the report)

| Model | Train F1 | Val F1 | Kaggle |
|---|---|---|---|
| EfficientNetV2 | 0.32 | 0.33 | 0.28 |
| ResNet18 | 0.32 | 0.27 | 0.29 |
| ResNet50 (unbalanced) | 0.52 | 0.40 | 0.36 |
| **ResNet50 (balanced)** | **0.47** | **0.41** | **0.4012** |
| ResNet152 (balanced) | 0.92 | 0.63 | 0.24 |

The submitted notebook is [`notebooks/BestResult.ipynb`](Second_Challenge/notebooks/BestResult.ipynb), which produced [`submissions/submission_eff.csv`](Second_Challenge/submissions/submission_eff.csv).

---

## Repository structure

```
First_Challenge/
├── notebooks/
│   ├── NaN_First_challenge.ipynb   # submitted notebook
│   ├── f10.9761/                   # earlier BiLSTM experiment
│   └── f10.97/                     # earlier BiLSTM experiment
└── report/AN2DL_Report.pdf
Second_Challenge/
├── AN2DL_Report.pdf
├── notebooks/BestResult.ipynb      # submitted notebook
└── submissions/submission_eff.csv
```

## Data and weights

The competition datasets and trained checkpoints (several GB) are **not** included in this repository. To re-run the notebooks, download the data from the course Kaggle competitions and place it under `dataset/` as expected by the data-loading cells. You will also need to adapt the Google Drive paths in the first cells.
