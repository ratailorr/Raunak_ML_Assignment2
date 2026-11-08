# README

# PCOS Classifier Lab – Five ML Models + Streamlit Dashboard

**M.Tech (AIML) · Machine Learning · Assignment 2 (BITS Pilani WILP)**

🔗 **Live Streamlit app:** `https://PCOSModelRaunakTailor.streamlit.app` 

---

## a. Problem statement

Polycystic Ovary Syndrome (PCOS) is an endocrine disorder that affects a large share of women of
reproductive age. It is diagnosed with the Rotterdam criteria, which combine irregular ovulation,
clinical or biochemical signs of excess androgen, and polycystic ovaries on ultrasound. Because these
signs are spread across a gynaecology consult, blood tests and a scan, diagnosis is often delayed.

**Objective:** build a binary classifier that predicts `PCOS` (`1` = PCOS, `0` = no PCOS) from routine
clinical, hormonal and ultrasound measurements. The following five models are trained, tuned and
compared on the **same** held-out test set:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbours
4. Gaussian Naive Bayes
5. Support Vector Machine

Each model is scored with six metrics: Accuracy, AUC, Precision, Recall, F1 and MCC. The saved models
are then served through an interactive Streamlit dashboard that loads them and scores any uploaded
test CSV **without retraining**.

---

## b. Dataset description

| Property | Value |
| --- | --- |
| Source | Kaggle – *Polycystic ovary syndrome (PCOS)* by Prasoon Kottarathil (data collected from 10 hospitals across Kerala, India) |
| Files used | `dataset/PCOS_data_without_infertility.xlsx` (sheet `Full_new`, 541 × 45) and `dataset/PCOS_infertility.csv` (541 × 6) |
| Task | Binary classification – target `PCOS (Y/N)` → renamed `PCOS` |
| Instances | **541** patients (requirement ≥ 500 ✔) |
| Features | **41** raw predictors (requirement ≥ 12 ✔) + 4 engineered features inside the pipeline |
| Class balance | 364 No-PCOS (67.3 %) · 177 PCOS (32.7 %) – imbalance of about 2 : 1 |
| Data quality issues | `'1.99.'` typo in *II beta-HCG*, `'a'` in *AMH*, 1 blank *Marriage Status*, 1 blank *Fast food*, an all-empty `Unnamed: 44` column, and implausible values (pulse 13/18 bpm, systolic BP 12 mmHg, FSH 5052, LH 2018, Vit-D3 6014) |

**Feature families (41 predictors)**

| Family | Columns |
| --- | --- |
| Anthropometry | Age, WeightKg, HeightCm, BMI, HipIn, WaistIn, WaistHipRatio |
| Vitals & blood | PulseBpm, RespRate, Haemoglobin, SystolicBP, DiastolicBP, BloodGroup (coded 11–18), RBS |
| Menstrual / reproductive | CycleType (2 = regular, 4 = irregular), PeriodDays, YearsMarried, Pregnant, Abortions |
| Hormones | BetaHCG_I, BetaHCG_II, FSH, LH, FSHtoLH, TSH, AMH, Prolactin, VitD3, Progesterone |
| Symptoms & lifestyle (0/1) | WeightGain, HairGrowth, SkinDarkening, HairLoss, Pimples, FastFood, RegExercise |
| Ultrasound | FollicleCountL, FollicleCountR, FollicleSizeL, FollicleSizeR, EndometriumMm |

**Cleaning and modelling steps (`pcos_prep.py` + `train_pipeline.py`)**

1. **Load and cross-check.** The workbook is merged with `PCOS_infertility.csv` on `Sl. No`. The PCOS
label agrees on all 541 rows, and the CSV is used only to fill blanks in the shared hormone
columns. Serial-number, file-number and empty columns are dropped.
2. **Header harmonisation.** Raw headers (inconsistent spacing, typos such as *“aborptions”*) are
mapped to PascalCase names. The app accepts **either** the raw Kaggle headers or the cleaned ones.
3. **Typo repair.** `'1.99.'` becomes `1.99` and `'a'` becomes NaN, leaving 3 missing cells in total.
No rows are dropped.
4. **Hold-out split.** Stratified **75 / 25** split with `random_state = 42`, giving **405 training
rows and 136 test rows** (PCOS share 32.8 % / 32.4 %). The 136 test rows are exported, with all 41
features plus `PCOS`, to **`test_data.csv`**.
5. **Steps inside every saved pipeline** (fitted on training folds only, so there is no leakage):
    - `ClinicalFeatureBuilder` adds `IrregularCycle`, `FollicleTotal` (L + R), `LHtoFSH` and
    `SymptomScore` (count of five Y/N symptoms), and drops `CycleType`.
    - `PercentileClipper` winsorises the continuous columns at the training 1st and 99th
    percentiles, which neutralises the impossible readings without hand-written rules.
    - `SimpleImputer` uses the median for continuous columns and the most frequent value for flags
    and blood group.
    - The scaler is a tuned hyper-parameter: `RobustScaler`, `PowerTransformer` (Yeo-Johnson) or
    `MinMaxScaler`.
    - Blood group is one-hot encoded (51 model inputs in total), then the classifier is applied.

---

## c. GitHub repository link

**Repository:** `https://github.com/<your-github-username>/<your-repo-name>` *(replace with your repo URL)*

```
pcos-classifier-lab/
├── app.py                     Streamlit dashboard (upload CSV → pick model → live metrics)
├── pcos_prep.py               shared cleaning code + custom sklearn transformers
├── train_pipeline.py          full training script (split, tune, evaluate, save)
├── requirements.txt           pinned dependencies for Streamlit Community Cloud
├── README.md
├── test_data.csv              held-out test set – 136 rows × (41 features + PCOS)
├── .streamlit/config.toml     dark theme
├── dataset/
│   ├── PCOS_data_without_infertility.xlsx
│   └── PCOS_infertility.csv
└── model/
    ├── logreg_pipeline.pkl    Logistic Regression  (full preprocessing + model)
    ├── dtree_pipeline.pkl     Decision Tree
    ├── knn_pipeline.pkl       K-Nearest Neighbours
    ├── gnb_pipeline.pkl       Gaussian Naive Bayes
    ├── svm_pipeline.pkl       Support Vector Machine
    └── scoreboard.json        hold-out metrics, best params, split metadata
```

**Reproduce locally**

```bash
pip install -r requirements.txt
python train_pipeline.py        # about 25 s; rewrites model/*.pkl, model/scoreboard.json, test_data.csv
streamlit run app.py
```

Each `.pkl` is a complete scikit-learn `Pipeline`, so the app only calls `predict` and
`predict_proba` and never refits. `pcos_prep.py` must be present next to `app.py` because the
custom transformers are referenced by the pickles.

---

## d. Models used

All five classifiers were tuned with `GridSearchCV` on the 405 training rows using
**RepeatedStratifiedKFold (5 folds × 3 repeats)** and **MCC** as the selection score. MCC was chosen
because accuracy flatters a model on a 2 : 1 imbalanced target. Every number below was computed on
the **136-row held-out test set** (`test_data.csv`; 92 No-PCOS, 44 PCOS). PCOS is the positive class
for Precision, Recall and F1, and the decision threshold is 0.5.

### Comparison table (held-out test set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | **0.9044** | 0.9405 | 0.8974 | 0.7955 | **0.8434** | **0.7779** |
| Decision Tree | 0.8971 | 0.9333 | 0.8750 | 0.7955 | 0.8333 | 0.7609 |
| KNN | 0.8750 | **0.9465** | **0.9355** | 0.6591 | 0.7733 | 0.7108 |
| Naive Bayes | 0.8897 | 0.9427 | 0.8085 | **0.8636** | 0.8352 | 0.7534 |
| Support Vector Machine | 0.8897 | 0.9224 | 0.9143 | 0.7273 | 0.8101 | 0.7434 |

*Bold = best in column.*

**Selected hyper-parameters and generalisation check**

| Model | Best settings (grid search) | CV-MCC (train folds) | Train acc. | Test acc. |
| --- | --- | --- | --- | --- |
| Logistic Regression | L1 penalty, C = 0.3, RobustScaler | 0.774 | 0.928 | 0.904 |
| Decision Tree | entropy, max_depth = 3, min_samples_leaf = 12 | 0.734 | 0.906 | 0.897 |
| KNN | k = 13, Manhattan (p = 1), distance weights, PowerTransformer | 0.734 | 1.000 | 0.875 |
| Naive Bayes (Gaussian) | var_smoothing = 0.05, PowerTransformer | 0.723 | 0.889 | 0.890 |
| Support Vector Machine | RBF, C = 1.0, gamma = 0.02, PowerTransformer | 0.771 | 0.958 | 0.890 |

**Error breakdown per model (test set counts)**

| Model | TN | FP | FN | TP |
| --- | --- | --- | --- | --- |
| Logistic Regression | 88 | 4 | 9 | 35 |
| Decision Tree | 87 | 5 | 9 | 35 |
| KNN | 90 | 2 | 15 | 29 |
| Naive Bayes | 83 | 9 | 6 | 38 |
| Support Vector Machine | 89 | 3 | 12 | 32 |

### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Best overall: highest Accuracy (0.904), F1 (0.843) and MCC (0.778), with only 4 false alarms. The L1 penalty kept just 20 of the 51 inputs. The strongest positive weights are `SymptomScore`, `FollicleTotal`, right-ovary follicle count and `IrregularCycle`, which mirror the Rotterdam criteria. That explains why a linear decision boundary works so well. The train/test accuracy gap is small (0.928 → 0.904), so it generalises well. Its main weakness is that it misses 9 of 44 PCOS cases (recall 0.795). |
| Decision Tree | A very compact tree (depth 3, 7 leaves) came a close second on MCC (0.761) and tied with LR on recall (0.795). About 96 % of its split importance comes from the two engineered features `FollicleTotal` (0.63) and `SymptomScore` (0.34), which shows that the feature engineering paid off. The shallow depth and minimum leaf size kept train and test accuracy close (0.906 / 0.897). AUC (0.933) is lower because a tree with 7 leaves can only produce a few distinct probability values. |
| KNN | Best AUC (0.947) and Precision (0.936, only 2 false positives), so its probability ranking is excellent. At the 0.5 threshold, however, it is the most conservative model: recall is only 0.659 (15 of 44 PCOS cases missed), which gives the lowest F1 (0.773) and MCC (0.711). Distance weighting gives a training accuracy of 1.000, a clear sign that it memorises the training data. The 51-dimensional space dilutes distances even after Yeo-Johnson scaling. |
| Naive Bayes | Highest Recall (0.864, only 6 PCOS cases missed) and second-best AUC (0.943). That makes it the best screening model when a missed diagnosis is costlier than a false alarm. Correlated inputs (left/right follicle counts and their total, BMI/weight, FSH/LH and its ratio) break the independence assumption and inflate its confidence. The result is the most false positives (9), and precision drops to 0.809. Yeo-Johnson scaling, which makes features closer to Gaussian, was preferred by the grid search. Train and test accuracy are almost identical (0.889 / 0.890), so it is the most stable model. |
| Support Vector Machine | Second-best CV-MCC (0.771) and high precision (0.914), but it recovered only 32 of 44 PCOS cases (recall 0.727). Its AUC (0.922) is the lowest because Platt-scaled probabilities on 405 rows are noisier than LR’s native probabilities. The train/test gap (0.958 → 0.890) is the second largest after KNN, which suggests the RBF kernel fits some noise that a linear boundary does not need. |
| **Overall, Winner for your dataset?** | **Logistic Regression** is the winner. It ranks first on Accuracy, F1 and MCC (the most balanced metric for an imbalanced target), also had the highest cross-validated MCC on the training folds (0.774), generalises with a small gap, and is the most interpretable model, since its coefficients map directly onto clinical criteria. **Naive Bayes** is the recommended alternative if the priority is to miss as few PCOS cases as possible (highest recall). **KNN** ranks patients best (highest AUC) but would need a lower decision threshold to be clinically useful. With only 44 positive test cases, one patient moves recall by about 2.3 points, so the top-four differences in AUC are within sampling noise. |

---

## How the dashboard covers the rubric

The dashboard (`app.py`) uses sidebar navigation with four pages (Model evaluator, Leaderboard, Patient simulator, Project
notes), a dark theme and interactive Plotly charts.

| Rubric item | Where it is in the app |
| --- | --- |
| **a. Dataset upload (CSV)** | Sidebar → *Step 1 · Test data*. Upload any test CSV, or click **Use bundled test_data.csv**. A **Download sample** button is also provided. Headers are auto-harmonised, and missing columns or an invalid target produce clear errors. |
| **b. Model selection dropdown** | *Model evaluator* page → **Classifier** select box listing all 5 models. The *Leaderboard* page compares all of them together. |
| **c. Display of evaluation metrics** | Six metric tiles (Accuracy, AUC, Precision, Recall, F1, MCC), computed live from the loaded `.pkl` with no retraining. The leaderboard shows a sortable table with the best value per metric highlighted, plus a grouped bar chart. |
| **d. Confusion matrix / classification report** | Interactive confusion-matrix heatmap (counts plus row %), full classification report, ROC curve(s), per-row predictions with a “misclassified only” filter, and a CSV download. |
| **Extra: Patient simulator** | Live sliders and pills for one hypothetical patient (body, cycle, symptoms, optional follicle counts and hormones). All five saved pipelines score it instantly and show their probabilities as bars against the 0.5 threshold and a typical-patient baseline. Blank inputs are imputed by each pipeline, so nothing is retrained. |

---

## Caveats

- All patients come from hospitals in one Indian state and there are only 541 of them, so it is
unknown how well these models would transfer to other populations.
- Follicle counts from the ultrasound are themselves a Rotterdam diagnostic criterion. Using them as
inputs makes the problem easier than genuine early screening, so the scores here are optimistic.
- With just 44 positive cases in the test set, a single patient shifts recall by about 2.3 points.
Treat gaps smaller than roughly 0.02 between models as noise.
- Built for a course assignment. It is **not** a medical device and must not be used to diagnose anyone.