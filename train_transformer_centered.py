#!/usr/bin/env python3
"""
Train Transformer-based segmentation classifier (CodeBERT / DistilRoBERTa)
on Centered dataset with automatic environment tuning.

✅ Supports Transformers >=4.46 (uses eval_strategy)
✅ Logs per-epoch metrics
✅ FP16 on GPU, smaller subset on CPU
"""
import os, json, csv, torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from packaging import version
import transformers

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------
DATA_DIR = "data_generated/centered"
OUT_DIR = "runs/transformer_centered"
LANG = os.environ.get("LANG_NAME", "python")
MODEL_NAME = os.environ.get("MODEL_NAME", "microsoft/codebert-base")
EPOCHS = int(os.environ.get("EPOCHS", "3"))
BATCH = int(os.environ.get("BATCH", "8"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[INFO] Training {MODEL_NAME} on {LANG.upper()} with batch={BATCH}, epochs={EPOCHS}")
print(f"[INFO] Device detected: {DEVICE}")
print(f"[INFO] Transformers version: {transformers.__version__}")

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

train_path = os.path.join(DATA_DIR, f"{LANG}_train.jsonl")
valid_path = os.path.join(DATA_DIR, f"{LANG}_valid.jsonl")
train_samples = load_jsonl(train_path)
valid_samples = load_jsonl(valid_path)

print(f"[INFO] Loaded {len(train_samples)} training and {len(valid_samples)} validation samples.")

#if DEVICE == "cpu":
train_samples = train_samples[:20000]
valid_samples = valid_samples[:5000]
print(f"[WARN] CPU detected → using reduced subset: {len(train_samples)} train / {len(valid_samples)} valid")

train_dataset = Dataset.from_dict({
    "text": [s["text"] for s in train_samples],
    "label": [int(s["label"]) for s in train_samples],
})
eval_dataset = Dataset.from_dict({
    "text": [s["text"] for s in valid_samples],
    "label": [int(s["label"]) for s in valid_samples],
})

# -----------------------------------------------------
# TOKENIZATION
# -----------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = train_dataset.map(tokenize_fn, batched=True)
eval_dataset = eval_dataset.map(tokenize_fn, batched=True)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# -----------------------------------------------------
# MODEL
# -----------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# -----------------------------------------------------
# METRICS
# -----------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    csv_path = os.path.join(OUT_DIR, "metrics_log.csv")
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(metrics)
    return metrics

# -----------------------------------------------------
# TRAINING ARGS (version-safe)
# -----------------------------------------------------
arg_kwargs = dict(
    output_dir=OUT_DIR,
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    num_train_epochs=EPOCHS,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    save_total_limit=2,
)

# version compatibility for Transformers ≥ 4.46
if version.parse(transformers.__version__) >= version.parse("4.46.0"):
    arg_kwargs["eval_strategy"] = "epoch"
else:
    arg_kwargs["evaluation_strategy"] = "epoch"

args = TrainingArguments(**arg_kwargs)

# -----------------------------------------------------
# TRAINER
# -----------------------------------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(os.path.join(OUT_DIR, f"transformer_{LANG}.pt"))
print(f"[DONE] Saved → {OUT_DIR}/transformer_{LANG}.pt")
print(f"[INFO] Per-epoch metrics logged → {OUT_DIR}/metrics_log.csv")
