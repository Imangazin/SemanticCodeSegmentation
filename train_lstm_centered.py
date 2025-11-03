#!/usr/bin/env python3
"""
Train BiLSTM (Centered) — final embedding-based version.

This model:
  • Uses centered 100-char windows around each newline (from data_generated/centered)
  • Learns character embeddings
  • Employs BiLSTM + dropout + max pooling
  • Computes full evaluation metrics each epoch
  • Automatically adjusts LR on plateau
  • Saves model and metadata for reproducibility
"""
import os, json, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
DATA_DIR = "data_generated/centered"
OUT_DIR = "runs/lstm_centered"
LANG = os.environ.get("LANG_NAME", "python")
BATCH = int(os.environ.get("BATCH", "128"))
EPOCHS = int(os.environ.get("EPOCHS", "5"))
LR = float(os.environ.get("LR", "0.001"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------
# DATASET
# -------------------------------------------------------
class CenteredDataset(Dataset):
    def __init__(self, path):
        self.samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        x = [ord(c) if ord(c) < 256 else 0 for c in s["text"]]
        y = float(s["label"])
        return torch.tensor(x, dtype=torch.long), torch.tensor([y], dtype=torch.float32)

# -------------------------------------------------------
# MODEL
# -------------------------------------------------------
class BiLSTMCentered(nn.Module):
    def __init__(self, vocab=256, emb=64, hid=128, drop=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(hid * 2, 1)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.lstm(emb)
        out = self.dropout(out)
        pooled, _ = torch.max(out, dim=1)   # max pooling
        return self.fc(pooled).squeeze(-1)

# -------------------------------------------------------
# UTILITIES
# -------------------------------------------------------
def make_loader(lang, split, bs, shuffle=False):
    path = os.path.join(DATA_DIR, f"{lang}_{split}.jsonl")
    ds = CenteredDataset(path)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=2, pin_memory=True), ds

def compute_pos_weight(ds):
    counts = Counter(int(s["label"]) for s in ds.samples)
    n0, n1 = counts.get(0, 1), counts.get(1, 1)
    return torch.tensor([n0 / n1], dtype=torch.float32)

def evaluate(model, loader):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float().cpu().numpy()
            labels = yb.cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(labels)
    preds_all = np.array(preds_all).flatten()
    labels_all = np.array(labels_all).flatten()
    acc = accuracy_score(labels_all, preds_all)
    prec, rec, f1, _ = precision_recall_fscore_support(labels_all, preds_all, average="binary", zero_division=0)
    return acc, prec, rec, f1

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    tr, ds_train = make_loader(LANG, "train", BATCH, True)
    va, ds_valid = make_loader(LANG, "valid", BATCH, False)
    pos_weight = compute_pos_weight(ds_train).to(DEVICE)
    print(f"[INFO] Class balance for {LANG}: {pos_weight.item():.2f}:1 (neg:pos)")

    model = BiLSTMCentered().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = 0
    for e in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for xb, yb in tqdm(tr, leave=False):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb.squeeze())
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(tr)

        acc, prec, rec, f1 = evaluate(model, va)
        print(f"Epoch {e}/{EPOCHS} | Train loss={avg_train_loss:.4f} | Val Acc={acc:.4f} P={prec:.4f} R={rec:.4f} F1={f1:.4f}")

        # Adaptive LR on plateau
        if e > 1 and f1 < best_f1 + 1e-3:
            for g in opt.param_groups:
                g["lr"] *= 0.5
            print(f"[LR DROP] Plateau detected, new LR={opt.param_groups[0]['lr']:.6f}")
        best_f1 = max(best_f1, f1)

        torch.save(model.state_dict(), os.path.join(OUT_DIR, f"lstm_centered_{LANG}_e{e}.pt"))

    final_path = os.path.join(OUT_DIR, f"lstm_centered_{LANG}.pt")
    torch.save(model.state_dict(), final_path)
    print(f"[DONE] Saved final model → {final_path}")

    # Metadata
    meta = {
        "language": LANG,
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "learning_rate": LR,
        "embedding_dim": 64,
        "hidden_dim": 128,
        "device": DEVICE,
        "architecture": "BiLSTM (Centered)",
        "window_chars": 100,
        "dropout": 0.3
    }
    with open(os.path.join(OUT_DIR, f"lstm_centered_{LANG}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
