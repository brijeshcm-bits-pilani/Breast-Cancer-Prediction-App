# Breast Cancer Prediction App

## a. Problem Statement

Digitized image of Fine-needle aspiration (FNA) of a breast mass, from
which nuclear morphology can be measured viz., radius, texture, perimeter, area,
smoothness, compactness, concavity, concave points, symmetry, and fractal
dimension. This project treats those measurements as a **binary
classification problem**: the data contains ten mean/standard-error/worst-case
measurements per sample (30 features total), the objective is to predict whether the underlying
mass is **malignant** or **benign**. Six classical/ensemble classifiers are
trained on the same dataset so that their diagnostic performance
can be compared head-to-head, and the best-performing model is surfaced
through an interactive Streamlit app.


## b. Dataset Description

We use publicly available WDBC (Wisconsin Diagnostic Breast Cancer) dataset in this project.

- **Data Source**: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), UCI Machine Learning Repository (donated 1995).
- **Number of Instances**: 569 (357 benign, 212 malignant).
- **Number of Features**: 30 real-valued numeric features (10 base measurements × mean, standard error, worst).
- **Target/Output**: Diagnosis — `M` (malignant, encoded 1) / `B` (benign, encoded 0).
- **Access method**: `model/train.py` fetches the dataset from the official `ucimlrepo` client (`fetch_ucirepo(id=17)`) **only if no local cache exists yet**, the first run writes `model/wdbc_raw_cache.csv`, and every subsequent run reads that cache instead of downloading from the UCI ML website everytime. If no outbound connection to the UCI archive is available on the first run, it automatically falls back to scikit-learn's bundled copy of the identical WDBC records so the pipeline never breaks in a restricted network. Delete `model/wdbc_raw_cache.csv` to force a fresh fetch.
- **Splitting data for training and testing**: stratified 80/20 train/test split, `random_state=42`. The 20% test split (114 rows) is exported as `test_data.csv` and is used for testing the model performance.


## c. Link to GitHub Repository:

`[https://github.com/brijeshcm-bits-pilani/Breast-Cancer-Prediction-App](https://github.com/brijeshcm-bits-pilani/Breast-Cancer-Prediction-App)`


## d. Models Used:

All six models are trained on identically scaled (`StandardScaler`) versions of the same 80/20 split. For complete training code, refer: `model/train.py`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.8948 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| kNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Random Forest (Ensemble) | 0.9649 | 0.9944 | 1.0000 | 0.9048 | 0.9500 | 0.9258 |



### Observations on model performance:

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strongest all round performer, near-perfect AUC (0.996) shows the classes are almost linearly separable once features are standardized. High precision means very few benign masses are wrongly flagged malignant. |
| Decision Tree | Weakest  of the six models. A single tree overfits the training split's specific splits and generalizes worse than the ensemble/linear alternatives, lowest AUC and MCC of the group. |
| kNN | Good performance once features are scaled (distance based, so scaling matters a lot here); slightly behind Logistic Regression and Random Forest models on recall, a few malignant cases sit close to benign neighbors in feature space. |
| Naive Bayes | Same raw accuracy as the Decision Tree, but has  much higher AUC (0.989), its probability *ranking* is very good even though its default 0.5 threshold isn't perfectly calibrated for this feature distribution (Gaussian NB, assumes feature independence, which doesn't fully hold here). |
| Random Forest (Ensemble) | Ties Logistic Regression for best  accuracy and posts **perfect precision** (zero false-positive malignant calls on this split) by averaging over 300 trees, which overcomes single Decision Tree's overfitting problem. |
| **Overall Winner for the dataset is:?** | **Logistic Regression**,  having the best AUC and F1 score while staying fully interpretable (coefficients maps directly to feature effects). Random Forest is at second position and slightly better in  precision. |



## Repository Structure and description of files:

```
onco-signal-lab/
├─── app.py                        # Streamlit front-end (loads model_bundle.joblib)
├─── requirements.txt.             # python package, requirement file
├─── README.md                     # This file
├─── test_data.csv                 # held-out 20% test split (114 rows)
└─── model/
    ├─── train.py                  # fetch-if-not-cached data + trains all 6 models
    ├─── wdbc_raw_cache.csv         # local data cache (auto-created, gitignored)
    └─── model_bundle.joblib        # scaler  +  feature order  +  5 models  +  metrics, one file
```


## How to run the script for training, and streamline app launch:

```bash
pip install -r requirements.txt
python model/train.py              # fetches data (or reads cache), trains models, writes test_data.csv
streamlit run app.py               # launches the diagnostic assistant
```

In the running app: upload `test_data.csv` from the sidebar, pick a model
from the drop down, and view live accuracy/AUC/precision/recall/F1/MCC, a
confusion matrix, a full classification report, and a six-way model
comparison table, all computed on your uploaded rows..
