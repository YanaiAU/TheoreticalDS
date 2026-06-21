# Summary Report — Movie Rating Prediction

## Problem

Predict whether a movie is **highly rated** using metadata and text:

| Label | Definition |
|-------|------------|
| `High_Rated = 1` | `Vote_Average ≥ 7.0` |
| `High_Rated = 0` | otherwise |

**Dataset:** `mymoviedb.csv` — ~9,800 real TMDB-style movie records.

## Data balance

| Class | Count | Share |
|-------|------:|------:|
| Low-rated (< 7.0) | 6,600 | 67.2% |
| High-rated (≥ 7.0) | 3,226 | 32.8% |

Imbalance ratio ≈ **2.05 : 1** (negative : positive).

## Feature space (why d = 23)

Metadata features for linear / tree models:

| Group | Count | Features |
|-------|------:|----------|
| Numeric | 3 | `log_popularity`, `log_vote_count`, `Release_Year` |
| Genre indicators | 12 | Top-12 genres → binary flags |
| Language indicators | 8 | Top-8 languages → binary flags |
| **Total d** | **23** | |

**VC dimension (LogReg / linear SVM):** d + 1 = **24** (hyperplanes in ℝ²³).

Text models use separate representations (TF-IDF sparse vectors, MiniLM 384-d embeddings).

## Models evaluated

| Category | Model | Role |
|----------|-------|------|
| Baseline | Uniform random | Lower bound |
| Baseline | Majority class | Naïve imbalanced baseline |
| Metadata | Logistic Regression (L2) | Linear predictor + regularization |
| Metadata | CatBoost | Boosting / nonlinear structured data |
| Language | TF-IDF + LogReg | Bag-of-words text baseline |
| Language | TF-IDF + Random Forest | Nonlinear text baseline |
| Language | LSTM | Deep sequence model (from scratch) |
| Language | MiniLM + LogReg | Pretrained semantic embeddings |

## Key results (held-out test set)

| Model | Accuracy | Balanced Acc | F1 | ROC-AUC | PR-AUC |
|-------|----------|--------------|-----|---------|--------|
| **CatBoost (metadata)** | **~0.79** | **~0.78** | **~0.70** | **~0.87** | **~0.78** |
| LogReg (metadata) | ~0.73 | ~0.72 | ~0.63 | ~0.80 | ~0.67 |
| MiniLM+LogReg (language) | ~0.68–0.72 | ~0.66 | ~0.55 | ~0.72 | ~0.55 |
| TF-IDF+LogReg (text) | ~0.65 | ~0.61 | ~0.49 | ~0.66 | ~0.49 |
| LSTM (text) | ~0.58 | ~0.58 | ~0.47 | ~0.61 | ~0.44 |

*Exact numbers depend on run; see notebook output table `final_model_results.csv`.*

## Main findings

1. **Metadata dominates.** Popularity and vote count strongly proxy audience reception; CatBoost reaches ~79% accuracy.
2. **Text alone is weaker.** Overviews describe plot, not ratings; language models stay below metadata performance.
3. **Pretrained language > raw LSTM.** MiniLM embeddings outperform LSTM trained from scratch on ~10k samples.
4. **Theory aligns with practice.** LogReg has low VC dim (24); PAC bounds are meaningful when n ≈ 7,860 ≫ VC. Higher-capacity models (CatBoost, RF) can show larger generalization gaps.

## Course topics covered

| Course topic | How we use it |
|--------------|---------------|
| PAC / VC dimension | VC = d+1 for linear models; PAC bound on LogReg |
| Linear predictors | Logistic Regression baseline |
| Regularization (L2) | `class_weight` + L2 penalty in LogReg |
| Model selection | Stratified train/test split, multi-metric evaluation |
| Boosting | CatBoost |
| Kernel / high capacity | RF, CatBoost capacity discussion |
| Text / embeddings | TF-IDF, MiniLM semantic vectors |
| Generalization gap | Train vs test error comparison |

## Limitations

- Selection bias: popular movies receive more votes.
- Temporal drift not modeled (random stratified split).
- Text is marketing language, not critic reviews.
- No hyperparameter search (kept reproducible and focused).

## Conclusion

Structured metadata is the strongest signal for predicting high TMDB ratings. The project demonstrates classical ML baselines, modern text methods, and explicit connections to learning theory — suitable as a Theoretical Data Science course final project.
