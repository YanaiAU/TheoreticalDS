# WikiShield

Detect **AI-generated Wikipedia-style intros** — human lead sections vs GPT text
on the same topics.

Course project for *Theoretical Foundations of Data Science* (Ariel University):
pick a real dataset, visualize, state preconceptions, compare models, ship on GitHub.

---

## Results (headline)

| Model | Accuracy | F1 (AI) | ROC-AUC |
|-------|----------|---------|---------|
| **Linear SVM (best)** | **0.968** | **0.968** | **0.995** |
| Logistic Regression | 0.965 | 0.965 | 0.995 |
| HistGradientBoosting | 0.928 | 0.927 | 0.980 |
| Random Forest | 0.904 | 0.901 | 0.966 |
| Complement NB | 0.886 | 0.880 | 0.948 |

- **Data:** 150k texts (75k Wikipedia topics × human + GPT), topic-held-out 120k/30k split  
- **Features:** TF-IDF (8k) + stylistic cues  
- Artifacts: `results/fig1…fig7`, `pickles/best_model.joblib`

---

## Why this problem

Wikipedia is a high-stakes knowledge surface. If LLMs quietly inject fluent but
wrong lead paragraphs, readers and downstream models absorb it. This project
treats that as a **trust & safety** text-classification task: score a paragraph
as human vs machine using classical features (TF-IDF + style stats) and several
learners — fast enough to run offline without a GPU.

---

## Dataset (real, large)

[`aadityaubhat/GPT-wiki-intro`](https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro)
on Hugging Face:

- **150k** Wikipedia topics
- Real Wikipedia introductions + GPT (Curie) introductions for the same titles
- Flattened to a binary table: `label=0` human, `label=1` AI
- Default download: **75k topics → ~150k texts** (use `--max-topics 150000` for the full ~300k)

No synthetic template generator. Human side is Wikipedia; machine side is real GPT output from the published corpus.

Train/test split is **held out by `topic_id`**, so a page’s human and AI twins never appear on opposite sides of the split.

---

## Layout

```
wikishield/
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   └── wiki_ai_detection.csv     <- after download (gitignored)
├── src/
│   ├── download_data.py
│   ├── features.py
│   ├── eda.py
│   ├── train.py
│   ├── plot_results.py
│   └── predict.py
├── pickles/                      <- joblib artifacts after train
├── results/                      <- figures + model_comparison.csv/json
└── notebooks/
    └── walkthrough.ipynb
```

---

## Quickstart

```bash
cd wikishield
pip install -r requirements.txt

python src/download_data.py
python src/eda.py data/wiki_ai_detection.csv
python src/train.py data/wiki_ai_detection.csv
python src/plot_results.py

python src/predict.py --text "Sexhow railway station was a railway station built to serve the hamlet of Sexhow in North Yorkshire, England."
```

Artifacts land in `pickles/` (`featurizer.joblib`, `best_model.joblib`, `test_preds.joblib`).
Metrics and figures land in `results/`.

---

## Models compared

| Model | Notes |
|-------|--------|
| Logistic Regression | Linear baseline, class-balanced |
| Complement NB | Strong sparse-text baseline |
| Calibrated Linear SVM | Margin classifier + probabilities |
| Random Forest | On TruncatedSVD(120) of TF-IDF+style |
| HistGradientBoosting | Same SVD projection |

Selection metric: **F1 on the AI class**, plus ROC-AUC / PR-AUC.

---

## Preconceptions (before looking at scores)

1. GPT intros may be shorter and more length-uniform than real Wikipedia leads.
2. Lexical diversity / punctuation cadence may shift under generation.
3. Same-topic pairing makes TF-IDF look very strong in-lab — real deployment would face unseen topics and newer LLMs.
4. Linear sparse models should carry most of the lift; trees matter if style metas help.

---

## Theory hooks (course)

- **Hypothesis class:** linear separators over TF-IDF ∪ style features → VC dimension scales with feature count (~8k + 9 style dims).
- **Generalization:** topic-held-out split approximates a domain shift across pages; same-topic leakage would inflate accuracy.
- **Imbalance:** this corpus is balanced 50/50 by construction; still report F1 / PR-AUC as if the positive class were rare in production.

---

## License / credit

Dataset: [GPT-wiki-intro](https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro) (Aaditya Bhat).  
Project code: for the course submission.
