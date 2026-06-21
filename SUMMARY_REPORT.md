# Summary Report — Movie Rating Prediction (Final)

## Goal
Predict whether a movie is **highly rated**:
- **Label**: `High_Rated = 1` if `Vote_Average >= 7.0`, else `0`
- **Dataset**: `mymoviedb.csv` (~9.8k rows)

## Data balance
- `< 7.0`: ~67%
- `>= 7.0`: ~33%
- Imbalance ratio: ~2.05:1 (negative:positive)

## Data preparation
- **Cleaning**:
  - Parsed `Release_Date`, extracted `Release_Year`
  - Coerced `Popularity`, `Vote_Count`, `Vote_Average` to numeric
  - Log transforms: `log(1+Popularity)`, `log(1+Vote_Count)`
  - Dropped rows with missing required fields for each experiment
- **Features**:
  - Metadata: `Release_Year`, `log_popularity`, `log_vote_count`, top genres, top languages
  - Text: `Overview` (TF‑IDF and optionally BERT)

## Experimental setup
- **Split**: train/test with `stratify=y`, random seed 42
- **Balanced training**: undersampling majority class (test set unchanged)
- **Metrics** (imbalance-aware):
  - ROC‑AUC, PR‑AUC (Average Precision)
  - F1, Balanced Accuracy
  - Confusion matrix at threshold 0.5 (or tuned threshold if used)

## Baselines
- **Uniform random**: predicts 0/1 with 50% probability
- **Majority class**: always predict `< 7.0`

## Models compared
### Metadata only
- Logistic Regression (L2, class_weight balanced)
- CatBoost (if available)

### Text only (Overview)
- TF‑IDF + Logistic Regression
- TF‑IDF + Random Forest

### Hybrid (metadata + language understanding)
- MiniLM sentence embeddings + CatBoost on `Overview` + metadata

## Results (paste table here)
Paste the final sorted table from `final_movie_project.ipynb`:

| Model | Accuracy | Balanced Acc | F1 | ROC‑AUC | PR‑AUC |
|------|----------|--------------|----|--------|--------|
| … | … | … | … | … | … |

## Key takeaways
- **Best overall model**:
- **Does text help?** (metadata vs text vs hybrid):
- **Error analysis**: examples of false positives/false negatives
- **Trade-offs**: accuracy vs interpretability vs compute

## Limitations
- Selection bias: popular movies receive more votes
- Possible temporal drift (older vs newer movies)
- Text field quality varies (short/long overviews, language)

## Future work (if time)
- One shared split for **all** models (strict comparability)
- Hybrid model: concatenate metadata + TF‑IDF (or stacking)
- Threshold tuning to optimize F1 or PR‑AUC

