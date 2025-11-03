#!/usr/bin/env python3
"""
Train Logistic Regression (BoC baseline)
- Uses Bag-of-Characters numeric matrices (7×256 = 1792 features)
- Reads from data_generated/boc/
- Handles class imbalance and skips invalid JSON lines
- Saves trained model (.joblib) and metadata (.json)
"""
import os, json, numpy as np, joblib
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from collections import Counter

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DATA_DIR = "data_generated/boc"
OUT_DIR = "runs/lr_boc"
LANG = os.environ.get("LANG_NAME", "python")
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def load_jsonl(path):
    X, y = [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Skipping invalid JSON at line {i} in {os.path.basename(path)}")
                continue
            if "X" in j and "y" in j:
                try:
                    arr = np.array(j["X"], dtype=np.float32).reshape(-1)
                    X.append(arr)
                    y.append(int(j["y"]))
                except Exception as e:
                    print(f"[WARN] Skipping malformed row {i}: {e}")
    if not X:
        raise ValueError(f"No valid samples found in {path}")
    return np.stack(X), np.array(y)

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    train_path = os.path.join(DATA_DIR, f"{LANG}_train.jsonl")
    valid_path = os.path.join(DATA_DIR, f"{LANG}_valid.jsonl")

    print(f"[INFO] Loading data for {LANG}...")
    X_train, y_train = load_jsonl(train_path)
    X_valid, y_valid = load_jsonl(valid_path)
    print(f"[INFO] {LANG}: train={X_train.shape[0]}, valid={X_valid.shape[0]}, features={X_train.shape[1]}")
    print(f"[INFO] Label distribution (train): {Counter(y_train)}")
    print(f"[INFO] Label distribution (valid): {Counter(y_valid)}")

    print("[INFO] Training Logistic Regression (Bag-of-Characters)...")
    model = LogisticRegression(
        max_iter=2000,
        solver="saga",
        class_weight="balanced",
        n_jobs=-1,
        C=2.0
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    acc = accuracy_score(y_valid, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_valid, y_pred, average="binary")

    print(f"\n[RESULT] {LANG} → Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}\n")
    print(classification_report(y_valid, y_pred, digits=4))

    # Save model
    job_path = os.path.join(OUT_DIR, f"lr_{LANG}.joblib")
    joblib.dump(model, job_path)
    print(f"[DONE] Saved model → {job_path}")

    # Save metadata for reproducibility
    meta = {
        "language": LANG,
        "features": int(X_train.shape[1]),
        "train_samples": int(X_train.shape[0]),
        "valid_samples": int(X_valid.shape[0]),
        "window_lines": 7,
        "char_space": 256,
        "model_type": "LogisticRegression-BoC",
        "solver": "saga",
        "class_weight": "balanced",
        "C": 2.0,
        "max_iter": 2000
    }
    meta_path = os.path.join(OUT_DIR, f"lr_{LANG}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[DONE] Saved metadata → {meta_path}")

if __name__ == "__main__":
    main()
