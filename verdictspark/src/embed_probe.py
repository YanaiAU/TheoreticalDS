"""
embed_probe.py — Qwen3 hidden-state embeddings + classical heads.

This is the quality winner path: bag-of-words fails on SCOTUS facts;
Qwen representations + a light probe usually lift balanced accuracy.
Also times embedding latency for the Pareto plot.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODEL = ROOT / "models" / "qwen3-0.6b"
MAX_LEN = 512
BATCH = 8


@torch.inference_mode()
def embed_texts(model, tok, texts: list[str], device: str) -> np.ndarray:
    vecs = []
    lat = []
    for i in tqdm(range(0, len(texts), BATCH), desc="embed"):
        batch = texts[i : i + BATCH]
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(**enc, output_hidden_states=True)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) / len(batch))
        hidden = out.hidden_states[-1]  # [B, T, H]
        mask = enc["attention_mask"].unsqueeze(-1)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        mean = summed / counts
        vecs.append(mean.float().cpu().numpy())
    X = np.vstack(vecs)
    return X, float(np.mean(lat) * 1000)


def metrics(y_true, y_pred, y_proba=None):
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_proba is not None:
        m["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return m


def main():
    train = pd.read_csv(RESULTS / "train_split.csv")
    test = pd.read_csv(RESULTS / "test_split.csv")
    # keep overnight-friendly but large enough
    rng = np.random.default_rng(42)
    # use up to 2000 train / full test (or 800 test)
    if len(train) > 2000:
        idx = []
        for lab in (0, 1):
            sub = train.index[train["label"] == lab].to_numpy()
            take = min(len(sub), 1000)
            idx.extend(rng.choice(sub, size=take, replace=False).tolist())
        train = train.loc[idx].reset_index(drop=True)
    if len(test) > 800:
        idx = []
        for lab in (0, 1):
            sub = test.index[test["label"] == lab].to_numpy()
            take = min(len(sub), 400)
            idx.extend(rng.choice(sub, size=take, replace=False).tolist())
        test = test.loc[idx].reset_index(drop=True)

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()

    print(f"Embedding train={len(train)} test={len(test)}")
    X_tr, _ = embed_texts(model, tok, train["text"].astype(str).tolist(), device)
    X_te, lat_ms = embed_texts(model, tok, test["text"].astype(str).tolist(), device)
    y_tr = train["label"].astype(int).to_numpy()
    y_te = test["label"].astype(int).to_numpy()

    # free VRAM before sklearn
    del model
    torch.cuda.empty_cache()

    probes = {
        "qwen06_logreg_probe": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "qwen06_mlp_probe": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 64),
                        max_iter=40,
                        random_state=42,
                        early_stopping=True,
                    ),
                ),
            ]
        ),
    }

    rows = []
    best_name, best_bal, best = None, -1.0, None
    for name, pipe in probes.items():
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        proba = pipe.predict_proba(X_te)[:, 1] if hasattr(pipe, "predict_proba") else None
        m = metrics(y_te, pred, proba)
        m["model"] = name
        m["family"] = "qwen_probe"
        m["latency_ms"] = lat_ms  # dominated by embedding pass
        rows.append(m)
        print(name, m)
        print(classification_report(y_te, pred, target_names=["respondent", "petitioner"], digits=3))
        if m["balanced_accuracy"] > best_bal:
            best_name, best_bal, best = name, m["balanced_accuracy"], pipe

    out = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    out.to_csv(RESULTS / "qwen_probe_comparison.csv", index=False)
    (RESULTS / "qwen_probe_meta.json").write_text(
        json.dumps(
            {
                "best": best_name,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "embed_latency_ms": lat_ms,
                "model": str(MODEL),
            },
            indent=2,
        )
    )
    print(out.to_string(index=False))
    print(f"Best probe: {best_name} bal={best_bal:.3f}")


if __name__ == "__main__":
    main()
