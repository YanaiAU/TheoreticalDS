"""Qwen3-0.6B mean-pool embeddings + probes on SST-5."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODEL = ROOT / "models" / "qwen3-0.6b"
RESULTS.mkdir(exist_ok=True)
MAX_LEN = 128
BATCH = 16


@torch.inference_mode()
def embed(model, tok, texts, device):
    vecs, lats = [], []
    for i in tqdm(range(0, len(texts), BATCH), desc="embed"):
        batch = texts[i : i + BATCH]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
        enc = {k: v.to(device) for k, v in enc.items()}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(**enc, output_hidden_states=True)
        torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) / len(batch))
        h = out.hidden_states[-1]
        mask = enc["attention_mask"].unsqueeze(-1)
        mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        vecs.append(mean.float().cpu().numpy())
    return np.vstack(vecs), float(np.mean(lats) * 1000)


def main():
    tr = pd.read_csv(DATA / "sst5_train.csv")
    te = pd.read_csv(DATA / "sst5_validation.csv")
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()

    X_tr, _ = embed(model, tok, tr["text"].astype(str).tolist(), device)
    X_te, lat = embed(model, tok, te["text"].astype(str).tolist(), device)
    y_tr = tr["label"].astype(int).to_numpy()
    y_te = te["label"].astype(int).to_numpy()
    del model
    torch.cuda.empty_cache()

    probes = {
        "qwen06_logreg_probe": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1, random_state=42)),
            ]
        ),
        "qwen06_mlp_probe": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(512, 256),
                        max_iter=80,
                        random_state=42,
                        early_stopping=True,
                    ),
                ),
            ]
        ),
    }

    rows = []
    best_name, best_f1 = None, -1.0
    for name, pipe in probes.items():
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        row = {
            "model": name,
            "family": "qwen_probe",
            "accuracy": float(accuracy_score(y_te, pred)),
            "macro_f1": float(f1_score(y_te, pred, average="macro")),
            "weighted_f1": float(f1_score(y_te, pred, average="weighted")),
            "latency_ms": lat,
        }
        rows.append(row)
        print(f"\n=== {name} ===")
        print(classification_report(y_te, pred, digits=3, zero_division=0))
        print(row)
        if row["macro_f1"] > best_f1:
            best_name, best_f1 = name, row["macro_f1"]

    cmp = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    cmp.to_csv(RESULTS / "qwen_probe_comparison.csv", index=False)
    (RESULTS / "qwen_probe_meta.json").write_text(
        json.dumps({"best": best_name, "embed_latency_ms": lat, "model": str(MODEL), "task": "sst5"}, indent=2)
    )
    print(cmp.to_string(index=False))
    print("Best Qwen probe:", best_name, "macro_f1=", best_f1)


if __name__ == "__main__":
    main()
