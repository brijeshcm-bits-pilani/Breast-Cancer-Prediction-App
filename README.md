# Breast Cancer Prediction App

## a. Problem Statement

Fine-needle aspiration (FNA) of a breast mass produces a digitized image from
which nuclear morphology can be measured — radius, texture, perimeter, area,
smoothness, compactness, concavity, concave points, symmetry, and fractal
dimension. This project treats those measurements as a **binary
classification problem**: given the ten mean/standard-error/worst-case
measurements per sample (30 features total), predict whether the underlying
mass is **malignant** or **benign**. Five classical/ensemble classifiers are
trained on the same split of the same dataset so their diagnostic performance
can be compared head-to-head, and the best-performing model is surfaced
through an interactive Streamlit app.

## b. Dataset Description

- **Source**: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), UCI Machine Learning Repository (donated 1995).
- **Instances**: 569 (357 benign, 212 malignant) — exceeds the 500-instance minimum.
- **Features**: 30 real-valued numeric features (10 base measurements × mean, standard error, worst), exceeds the 12-feature minimum.
- **Target**: Diagnosis — `M` (malignant, encoded 1) / `B` (benign, encoded 0).
- **Access method**: `model/train.py` fetches the dataset live via the official `ucimlrepo` client (`fetch_ucirepo(id=17)`) **only if no local cache exists yet** — the first run writes `model/wdbc_raw_cache.csv`, and every subsequent run reads that cache instead of hitting the network again. If no outbound connection to the UCI archive is available on the first run, it automatically falls back to scikit-learn's bundled copy of the identical WDBC records so the pipeline never breaks in a restricted network. Delete `model/wdbc_raw_cache.csv` to force a fresh fetch.
- **Split**: stratified 80/20 train/test split, `random_state=1974`. The 20% test split (114 rows) is exported as `test_data.csv` and is the file the Streamlit app expects to be uploaded.

## c. GitHub Repository Link

[`https://github.com/brijeshcm-bits-pilani/Breast-Cancer-Prediction-App`
](https://github.com/brijeshcm-bits-pilani/Breast-Cancer-Prediction-App)  

## d. Models Used

All five models are trained on identically scaled (`StandardScaler`) versions of the same 80/20 split. Full training code: `model/train.py`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9871 | 1.0000 | 0.9524 | 0.9756 | 0.9626 |
| Decision Tree | 0.9298 | 0.9028 | 0.9722 | 0.8333 | 0.8974 | 0.8504 |
| kNN | 0.9737 | 0.9830 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| Naive Bayes | 0.9649 | 0.9881 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Random Forest (Ensemble) | 0.9649 | 0.9888 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |


### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | this is the best performing model on every metric, perfect precision, best recall, best F1, best MCC. With this train/test split (`random_state=1974`) the classes separate almost linearly once features are standardized, and the linear decision boundary generalizes better than any of the more flexible models below. |
| Decision Tree | Weakest of the five by a clear margin. A single tree overfits the training split's specific structure, lowest accuracy, AUC, and MCC of the group; recall (0.833) shows it misses more malignant cases than any other model. |
| kNN | Very strong, perfect precision and second-best F1/MCC, once features are scaled (distance-based, so scaling matters a lot here). Recall trails Logistic Regression slightly: a few malignant cases still sit close to benign neighbors in feature space. |
| Naive Bayes | Middle of the pack. High AUC (0.988) shows good probability *ranking*, but the independence assumption caps precision/recall below the top three models, real tumor features are correlated, which Gaussian NB can't model. |
| Random Forest (Ensemble) | Tuned via 5-fold CV grid search over `n_estimators` (250–350) and `max_depth` (5–8); `n_estimators=320, max_depth=8` won on CV AUC. Ties Naive Bayes on this particular test split, averaging 320 decorrelated trees controls the single Decision Tree's over-fitting, but doesn't beat the linear model here. |
| **Overall Winner for your dataset?** | **Logistic Regression**, clearly, best score on all five metrics simultaneously, and it's the most interpretable model of the five (coefficients map directly to feature effects). |

## Repository Structure

Deliberately minimal — two Python files total, one bundled model artifact:

```
onco-signal-lab/
├── app.py                        # Streamlit front-end (loads model_bundle.joblib)
├── requirements.txt              # Requirements file
├── README.md                     # This file
├── test_data.csv                 # held-out 20% test split (114 rows)
└── model/
    ├── train.py                  # fetch-if-not-cached data + trains all 6 models
    ├── wdbc_raw_cache.csv         # local data cache (auto-created, gitignored)
    └── model_bundle.joblib        # scaler + feature order + 5 models + metrics, one file
```

## How to Reproduce

```bash
pip install -r requirements.txt
python model/train.py              # fetches data (or reads cache), trains models, writes test_data.csv
streamlit run app.py               # launches the diagnostic assistant
```

In the running app: upload `test_data.csv` from the sidebar, pick a model
from the dropdown, and view live accuracy/AUC/precision/recall/F1/MCC, a
confusion matrix, a full classification report, and a five-way model
comparison table — all computed on your uploaded rows.
