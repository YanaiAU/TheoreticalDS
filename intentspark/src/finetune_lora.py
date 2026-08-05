"""LoRA fine-tune Qwen3-0.6B for SST-5 (the quality winner)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "qwen3-0.6b"
RESULTS = ROOT / "results"
OUT = ROOT / "pickles" / "qwen_lora_sst5_adapter"
RESULTS.mkdir(exist_ok=True)


def main():
    tr = pd.read_csv(ROOT / "data" / "sst5_train.csv")
    va = pd.read_csv(ROOT / "data" / "sst5_validation.csv")
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    def tok_map(batch):
        o = tok(batch["text"], truncation=True, max_length=128, padding="max_length")
        o["labels"] = batch["label"]
        return o

    ds_tr = Dataset.from_pandas(tr[["text", "label"]]).map(tok_map, batched=True, remove_columns=["text", "label"])
    ds_va = Dataset.from_pandas(va[["text", "label"]]).map(tok_map, batched=True, remove_columns=["text", "label"])
    ds_tr.set_format("torch")
    ds_va.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL), num_labels=5, dtype=torch.bfloat16, trust_remote_code=True
    )
    model.config.pad_token_id = tok.pad_token_id
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.print_trainable_parameters()
    model.cuda()

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, pred)),
            "macro_f1": float(f1_score(labels, pred, average="macro")),
        }

    args = TrainingArguments(
        output_dir=str(ROOT / "pickles" / "qwen_lora_sst5"),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-4,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print("EVAL", metrics)

    preds = []
    model.eval()
    for i in range(0, len(va), 32):
        batch = va["text"].astype(str).tolist()[i : i + 32]
        enc = tok(batch, return_tensors="pt", truncation=True, max_length=128, padding=True)
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(logits.argmax(-1).cpu().tolist())
    y = va["label"].to_numpy()
    pred = np.asarray(preds)

    # latency
    enc = tok(va["text"].astype(str).tolist()[:64], return_tensors="pt", truncation=True, max_length=128, padding=True)
    enc = {k: v.cuda() for k, v in enc.items()}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(**enc)
    torch.cuda.synchronize()
    lat = (time.perf_counter() - t0) / 64 * 1000

    row = {
        "model": "qwen06_lora_finetune",
        "family": "qwen_finetuned",
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "weighted_f1": float(f1_score(y, pred, average="weighted")),
        "latency_ms": float(lat),
    }
    print(row)
    print(classification_report(y, pred, digits=3, zero_division=0))
    pd.DataFrame([row]).to_csv(RESULTS / "qwen_lora_comparison.csv", index=False)
    (RESULTS / "qwen_lora_meta.json").write_text(json.dumps(row, indent=2))
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT))
    tok.save_pretrained(str(OUT))
    print("Saved adapter ->", OUT)


if __name__ == "__main__":
    main()
