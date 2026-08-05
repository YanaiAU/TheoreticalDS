# IntentSpark → SST-5 Sentiment

**Fine-grained movie-review sentiment (5 classes)** on real Stanford Sentiment Treebank data.

Classical TF-IDF models struggle. **Qwen3-0.6B + LoRA fine-tune** wins by a wide margin.
Linear SVM is **not** the best model.

Dataset: [`SetFit/sst5`](https://huggingface.co/datasets/SetFit/sst5)  
(Stanford Sentiment Treebank, 5-way: very negative → very positive)

---

## Results (validation set)

| Model | Accuracy | Macro-F1 | Notes |
|-------|---------:|---------:|-------|
| Linear SVM (TF-IDF) | 0.358 | 0.348 | collapses |
| Complement NB (TF-IDF) | 0.391 | 0.361 | |
| Logistic Regression (TF-IDF) | 0.383 | 0.380 | best classical |
| Qwen3-0.6B frozen embed + MLP | 0.434 | 0.416 | better, still limited |
| **Qwen3-0.6B LoRA fine-tune** | **0.513** | **0.501** | **winner** |

Chance accuracy ≈ **0.20**.  
**+13 accuracy points / +12 macro-F1** vs best TF-IDF LogReg.  
**+15 accuracy / +15 macro-F1** vs Linear SVM.

Figures: `results/fig1_macro_f1.png`, `fig2_accuracy.png`, `fig3_pareto.png`

---

## Why this works as a course project

1. **Real data** (SST-5 / Stanford)  
2. **Hard 5-class** problem — bag-of-words looks bad for a reason  
3. **Obvious lift** from fine-tuned Qwen (not a 0.3% nothing-burger)  
4. Linear SVM loses — Qwen wins  
5. Reproducible scripts + plots  

---

## Reproduce

```bash
cd intentspark
pip install -r requirements.txt peft
python src/download_data.py
python src/classical.py
python src/embed_probe.py
# LoRA fine-tune (GPU, a few minutes):
python src/finetune_lora.py
python src/plot_results.py
```

Local model: `models/qwen3-0.6b` (HF Qwen3-0.6B junction).
