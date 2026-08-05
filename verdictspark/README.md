# VerdictSpark

**Predict who wins a U.S. Supreme Court case from the facts** — petitioner vs
respondent — and show why bag-of-words linear models fail on hard legal text
while a **Qwen3** representation wins.

Course project · Theoretical Foundations of Data Science (Ariel University)

---

## Results (the point)

| Model | Balanced acc | Macro-F1 | Latency |
|-------|-------------:|---------:|--------:|
| Majority baseline | 0.500 | 0.394 | ~0 ms |
| Linear SVM (TF-IDF) | 0.500 | 0.394 | ~0.16 ms |
| Complement NB | 0.507 | 0.409 | ~0.13 ms |
| Logistic Regression (TF-IDF) | 0.518 | 0.518 | ~0.14 ms |
| Qwen3-0.6B generative | 0.515 | 0.395 | ~139 ms |
| Qwen3-0.6B + **speculative** decoding | 0.515 | 0.395 | **~80 ms** |
| **Qwen3-0.6B embedding + MLP probe** | **0.570** | **0.570** | ~13 ms |

**Winner is not Linear SVM.** Best quality: **Qwen MLP probe**.  
Speculative decoding keeps generative answers identical while cutting latency
~1.7× (139 → 80 ms/case).

Raw accuracy is a trap here (petitioner wins ~65% of cases). We optimize /
report **balanced accuracy**.

---

## Problem

Legal judgment prediction from clerk-written facts is long-domain NLP.
Dataset: [`drgary/ft9_justice`](https://huggingface.co/datasets/drgary/ft9_justice)
(~3.3k real SCOTUS cases from Oyez).

Classical TF-IDF models collapse toward the majority class. An LLM encoder
extracts semantics bag-of-words miss; a small probe on those embeddings lifts
balanced accuracy. Generative prompting is weaker on this task but is where
**speculative decoding** shows a clean speed win at matched quality.

---

## Layout

```
verdictspark/
├── README.md
├── models/qwen3-0.6b   <- local HF Qwen (junction)
├── models/qwen3-4b
├── models/dspark-qwen3-4b
├── vendor/DeepSpec     <- DSpark stack from Hector
├── data/scotus_justice.csv
├── pickles/
├── results/            <- metrics + figures
└── src/
    ├── download_data.py
    ├── classical.py
    ├── embed_probe.py      <- quality winner
    ├── llm_judge.py        <- generative ± speculative
    ├── plot_results.py
    └── run_overnight.py
```

---

## Reproduce

```bash
cd verdictspark
pip install -r requirements.txt

python src/download_data.py
python src/classical.py
python src/embed_probe.py
python src/llm_judge.py --mode vanilla --n 200
python src/llm_judge.py --mode speculative --n 200
python src/plot_results.py
```

GPU: RTX-class CUDA. Models load via Hugging Face from `models/`.

---

## Preconceptions (checked)

1. Majority baseline looks strong on raw accuracy — useless on balanced metrics. **Confirmed.**
2. TF-IDF linear/SVM models stay near chance on balanced accuracy. **Confirmed.**
3. Qwen representations beat bag-of-words. **Confirmed (MLP probe).**
4. Speculative decoding preserves labels and reduces latency. **Confirmed (~1.7×).**
