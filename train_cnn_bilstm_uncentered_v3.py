#!/usr/bin/env python3
"""
Train CNN-BiLSTM hybrid model (Uncentered dataset) — v3
Improvements:
 • Oversamples positive/boundary-rich snippets
 • Weighted BCE loss (manual reduction)
 • Larger sequence window (more context)
 • Lower learning rate for stability
 • Logs Precision/Recall/F1 per epoch
"""
import os, json, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from collections import Counter
import numpy as np

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
DATA_DIR = "data_generated/uncentered"
OUT_DIR = "runs/cnn_bilstm_uncentered_v3"
LANG = os.environ.get("LANG_NAME", "python")
BATCH = int(os.environ.get("BATCH", "64"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
LR = float(os.environ.get("LR", "2e-4"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------
# DATASET
# -------------------------------------------------------
class UncenteredDataset(Dataset):
    def __init__(self, path):
        self.samples = [json.loads(line) for line in open(path, encoding="utf-8")]
        # tag samples that contain at least one boundary
        self.weights = torch.tensor([
            3.0 if any(lbl == 1 for lbl in s["labels"]) else 1.0
            for s in self.samples
        ], dtype=torch.float32)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = [ord(c) if ord(c) < 256 else 0 for c in s["text"]]
        y = s["labels"]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.float32)

# -------------------------------------------------------
# MODEL
# -------------------------------------------------------
class CNNBiLSTM(nn.Module):
    def __init__(self, vocab=256, emb=64, hid=128, cnn_filters=128, drop=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.conv1 = nn.Conv1d(emb, cnn_filters, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=7, padding=3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop)
        self.lstm = nn.LSTM(cnn_filters, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid * 2, 1)

    def forward(self, x):
        x = self.emb(x).transpose(1, 2)           # [B, emb, seq]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.transpose(1, 2)                     # [B, seq, feat]
        out, _ = self.lstm(x)
        out = self.dropout(out)
        logits = self.fc(out).squeeze(-1)         # [B, seq]
        return logits

# -------------------------------------------------------
# UTILITIES
# -------------------------------------------------------
def make_loader(lang, split, bs, shuffle=False):
    path = os.path.join(DATA_DIR, f"{lang}_{split}.jsonl")
    ds = UncenteredDataset(path)
    sampler = WeightedRandomSampler(ds.weights, len(ds.weights), replacement=True) if shuffle else None
    loader = DataLoader(ds, batch_size=bs, sampler=sampler, shuffle=False if sampler else shuffle,
                        num_workers=2, pin_memory=True)
    return loader, ds

def compute_pos_weight(ds):
    counts = Counter()
    for s in ds.samples:
        for y in s["labels"]:
            counts[int(y)] += 1
    n0, n1 = counts.get(0, 1), counts.get(1, 1)
    return torch.tensor([n0 / n1], dtype=torch.float32)

def f1_score_torch(y_true, y_pred):
    eps = 1e-7
    tp = ((y_true == 1) & (y_pred == 1)).sum().float()
    fp = ((y_true == 0) & (y_pred == 1)).sum().float()
    fn = ((y_true == 1) & (y_pred == 0)).sum().float()
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    return f1.item(), prec.item(), rec.item()

def run_epoch(model, loader, opt, loss_fn, train=True):
    model.train(train)
    total_loss, preds_all, labels_all = 0, [], []
    for xb, yb in tqdm(loader, leave=False):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        if train: opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb).mean()  # manual mean
        if train:
            loss.backward()
            opt.step()
        total_loss += loss.item()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        preds_all.append(preds.cpu())
        labels_all.append(yb.cpu())
    preds_all = torch.cat(preds_all)
    labels_all = torch.cat(labels_all)
    f1, p, r = f1_score_torch(labels_all, preds_all)
    return total_loss / len(loader), f1, p, r

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    tr, ds_train = make_loader(LANG, "train", BATCH, True)
    va, ds_valid = make_loader(LANG, "valid", BATCH, False)
    pos_weight = compute_pos_weight(ds_train).to(DEVICE)
    print(f"[INFO] Class balance for {LANG}: {pos_weight.item():.2f}:1 (neg:pos)")

    model = CNNBiLSTM().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    best_f1 = 0
    for e in range(1, EPOCHS + 1):
        tl, tf1, tp, trc = run_epoch(model, tr, opt, loss_fn, True)
        vl, vf1, vp, vr = run_epoch(model, va, opt, loss_fn, False)
        print(f"Epoch {e}/{EPOCHS} | Train loss={tl:.4f}, F1={tf1:.4f} | "
              f"Val loss={vl:.4f}, F1={vf1:.4f}, P={vp:.4f}, R={vr:.4f}")
        if vf1 > best_f1:
            best_f1 = vf1
            torch.save(model.state_dict(),
                       os.path.join(OUT_DIR, f"cnn_bilstm_uncentered_{LANG}.pt"))
    print(f"[DONE] Best Val F1={best_f1:.4f} | Model saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
