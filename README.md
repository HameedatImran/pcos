# PCOS Diagnosis Using Machine Learning
A machine learning project predicting Polycystic Ovary Syndrome (PCOS) diagnosis 
from clinical, hormonal, and anthropometric features, designed to support early, 
scalable identification of at-risk women in resource-limited settings.

---

## Clinical Context
Polycystic Ovary Syndrome affects an estimated **8–13% of women of reproductive age** 
globally, yet remains significantly underdiagnosed — often identified only after years 
of symptom mismanagement. Traditional diagnosis relies on the **Rotterdam Criteria (2003)**, 
requiring at least two of: oligo/anovulation, clinical hyperandrogenism, and polycystic 
ovaries on ultrasound. This process demands specialist involvement and is frequently 
inaccessible in low-resource settings.

This project explores whether routinely collectable clinical data — hormonal panels, 
anthropometric measurements, and symptom profiles — can support reliable, early 
classification of PCOS using machine learning.

---

## Dataset
**Source:** Kaggle — PCOS Dataset Without Infertility  
**Size:** 541 patients × 44 features  
**Target:** `PCOS (Y/N)` — Binary classification (1 = PCOS positive, 0 = PCOS negative)  
**Class distribution:** 364 No PCOS (67%) · 177 PCOS (33%)

**Notable data quality issues addressed:**
- `BMI`, `FSH/LH`, and `Waist:Hip Ratio` stored as Excel `#NAME?` formula errors → recalculated from source columns
- Stray text entry (`'a'`) in `AMH(ng/mL)` → coerced to NaN and median-imputed
- Blood group encoding (11–18) inferred from value frequency distributions
- Cycle encoding (2 = Regular, 4/5 = Irregular) inferred from column name and PCOS crosstab

---

## Project Pipeline

### 1. Exploratory Data Analysis
- Univariate distributions for all 44 features (histograms, bar charts)
- Bivariate analysis — violin plots for all continuous features vs PCOS status
- Symptom prevalence comparison by PCOS status
- Full correlation heatmap + target correlation ranking

### 2. Preprocessing
- Dropped clinically irrelevant columns: `Pregnant(Y/N)`, `No. of abortions`, both beta-HCG columns, `Vit D3`, `PRG`, `Hb`, `Marriage Status`, lifestyle features
- Dropped source columns after recalculating derived features (`Weight`, `Height`, `Waist`, `Hip`)
- Dropped rows for 4 missing values
- Cycle regularity encoded as binary (0 = Regular, 1 = Irregular)

### 3. Feature Engineering
| Feature | Description |
|---|---|
| `Total_Follicles` | Sum of left + right ovarian follicle counts |
| `LH/FSH` | Elevated ratio is a hallmark of PCOS |
| `AMH_x_Follicles` | Interaction term — both elevated in PCOS |
| `Symptom_Score` | Count of androgenic symptoms (hirsutism, acne, hair loss, weight gain, skin darkening) |
| `BMI_Category` | WHO cutoffs: underweight / normal / overweight / obese |
| Log transforms | Applied to `LH`, `AMH`, `TSH`, `PRL`, `RBS` to reduce right skew |

### 4. Class Imbalance
**SMOTE** (Synthetic Minority Oversampling Technique) applied to training set only — 
minority class (PCOS) oversampled to match majority class before model fitting.

### 5. Modelling
Two models trained and compared:

| Model | Configuration |
|---|---|
| Logistic Regression | `class_weight='balanced'`, `max_iter=1000` |
| XGBoost | `RandomizedSearchCV` (30 iterations, 5-fold CV), `sample_weight='balanced'` |

**Best XGBoost parameters:** `n_estimators=200`, `max_depth=7`, `learning_rate=0.2`, `gamma=0.3`, `colsample_bytree=0.8`, `subsample=1.0`

---

## Results

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| Accuracy | 0.84 | **0.89** |
| Precision (PCOS) | 0.72 | **0.81** |
| Recall (PCOS) | 0.83 | **0.86** |
| F1 (PCOS) | 0.77 | **0.83** |
| ROC-AUC | 0.937 | 0.936 |

**XGBoost is the best model** across accuracy, precision, recall, and F1.  
Both models achieve near-identical ROC-AUC (~0.937), indicating comparable class separation 
— XGBoost makes better threshold-level decisions.

> Recall is prioritised over accuracy in this clinical context: a missed PCOS diagnosis 
> (false negative) carries greater harm than a false positive that prompts further testing.

---

## Top 10 Features (XGBoost Importance)

1. **Total antral follicle count** — morphological hallmark of PCOS; Rotterdam criterion
2. **Menstrual cycle regularity** — oligo/anovulation; Rotterdam criterion
3. **Skin darkening** — acanthosis nigricans; marker of insulin resistance
4. **Excessive hair growth** — hirsutism; clinical hyperandrogenism marker
5. **Cycle length (days)** — quantifies severity of menstrual irregularity
6. **Endometrial thickness** — thickened from chronic unopposed oestrogen exposure
7. **Weight gain** — central obesity exacerbates insulin resistance in PCOS
8. **Random blood sugar** — reflects glucose dysregulation and insulin resistance
9. **FSH/LH ratio** — characteristically *low* in PCOS (elevated LH relative to FSH)
10. **TSH** — differential diagnosis marker; hypothyroidism mimics PCOS symptoms

---

## Project Structure
```
pcos-diagnosis/
├── data/
│   └── PCOS_data_without_infertility.csv
├── models/
│   ├── pcos_model.sav          ← saved XGBoost model
│   └── pcos_scaler.pkl         ← fitted StandardScaler
├── outputs/
│   └── visualisations/         ← all generated plots
└── pcos-diagnosis-using-logistic-regression.ipynb
```

---

## Setup
```bash
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
scipy
```

---

## Tools & Libraries
Python · pandas · NumPy · scikit-learn · imbalanced-learn · XGBoost · matplotlib · seaborn · scipy

---

## Limitations
- Dataset sourced from a single centre (India) — generalisability to African populations requires validation
- Ultrasound-derived features (follicle counts, endometrial thickness) may not be available at all primary care levels
- Test set is small (n=108) — results should be interpreted with caution pending external validation
- Model is diagnostic support only — not a replacement for clinical assessment
