# TrafficSpark → Israel Road-Accident Severity

Predict whether a traffic accident is **serious** (fatal or severe injury) vs **light**, from scene / road / time features.

**Real government data:** Israel CBS Public Use File, police-reported accidents with casualties, **2020–2024** (~49,941 rows).

Source: [data.gov.il – PUF 2020–2024]([https://data.gov.il/dataset/02789da8-7a3e-4bfc-b771-1732b1cf403c](https://data.gov.il/he/datasets/lamas/2023-puf)) · [govil.ai overview](https://govil.ai/datasets/02789da8-7a3e-4bfc-b771-1732b1cf403c/)

**HistGradientBoosting wins.** Linear SVM is not the best model.

---

## Results (stratified 20% holdout)

| Model | Accuracy | Balanced acc | Macro-F1 | F1 (serious) | ROC-AUC |
|-------|---------:|-------------:|---------:|-------------:|--------:|
| Logistic Regression (OHE) | 0.717 | 0.690 | 0.667 | 0.537 | 0.754 |
| Linear SVM (OHE) | 0.723 | 0.693 | 0.671 | 0.541 | 0.754 |
| Random Forest | 0.796 | 0.644 | 0.665 | 0.455 | 0.760 |
| **HistGradientBoosting** | **0.745** | **0.703** | **0.688** | **0.555** | **0.770** |

Base rate ≈ **26% serious** / 74% light. Chance accuracy ≈ 0.74 if always predicting light — so **balanced accuracy / macro-F1 / AUC** matter more than raw accuracy.

Figures: `results/fig0_eda.png`, `fig1_macro_f1.png`, `fig2_auc.png`, `fig3_pareto.png`

---

## What we cleaned / engineered

1. Merged yearly PUF tables (2020–2024) via CKAN datastore API  
2. Target from `HUMRAT_TEUNA`: **1,2 → serious**, **3 → light**  
3. Dropped rows missing severity  
4. Features = road type, district/locality, hour/day/month, accident type, lighting, weather, road surface, speed limit, geometry (`X`,`Y`), …  
5. Categorical missing → `__MISSING__`; numeric → median impute  
6. Classical models: one-hot; trees: ordinal codes (handle high-cardinality locality better)  
7. `class_weight='balanced'` (trees / linear) because of imbalance  
8. Stratified train/test split (seed 42)

---

## Why this project works

1. **Local, real** Israeli open data (not a toy UCI set)  
2. Clear public-safety question: can scene features flag serious crashes?  
3. Imbalance forces honest metrics (macro-F1 / balanced acc / AUC)  
4. **Nonlinear tree model beats linear baselines** on the metrics that matter  
5. Reproducible scripts + plots + saved pickle  

### Preconceptions vs data

- Expect urban / pedestrian / night patterns to raise serious risk — models use those codes (`SUG_TEUNA`, `YOM_LAYLA`, `SUG_DEREH`, …).  
- Linear models underfit interactions (road type × hour × district). HistGB captures them.  
- Random Forest gets high accuracy by favoring the majority class; HistGB is better calibrated on the serious class.

---

## Layout

```
trafficspark/
  data/                 # combined CSV (+ raw yearly)
  src/
    download_data.py
    train_models.py
    plot_results.py
  results/              # metrics CSV, meta.json, figures
  pickles/              # best_model.joblib
  requirements.txt
```

---

## Reproduce

```bash
cd trafficspark
pip install -r requirements.txt
python src/download_data.py    # if data/ missing
python src/train_models.py
python src/plot_results.py
```

Course: Theoretical Foundations of Data Science · Ariel University.
