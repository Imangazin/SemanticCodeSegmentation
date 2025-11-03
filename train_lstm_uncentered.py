#!/usr/bin/env python3
"""
Train BiLSTM (Uncentered) model on 100-char windows.
Each window has per-character binary labels (newline segmentation).
"""
import os, json, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DATA_DIR = "data_generated/uncentered"
OUT_DIR = "runs/lstm_uncentered"
LANG = os.environ.get("LANG_NAME", "python")
BATCH = int(os.environ.get("BATCH", "64"))
EPOCHS = int(os.environ.get("EPOCHS", "3"))
LR = float(os.environ.get("LR", "1e-3"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------
# DATASET
# --------------------------------------------------
class UncenteredDataset(Dataset):
    def __init__(self, path):
        self.samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                self.samples.append(j)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        x = [ord(c) if ord(c) < 256 else 0 for c in s["text"]]
        y = s["labels"]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.float32)

# --------------------------------------------------
# MODEL
# --------------------------------------------------
class BiLSTM(nn.Module):
    def __init__(self, vocab=256, emb=32, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid * 2, 1)
    def forward(self, x):
        out, _ = self.lstm(self.emb(x))
        out = self.fc(out).squeeze(-1)
        return out

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def make_loader(lang, split, bs, shuffle=False):
    path = os.path.join(DATA_DIR, f"{lang}_{split}.jsonl")
    ds = UncenteredDataset(path)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=2, pin_memory=True)

def run_epoch(model, loader, opt, loss_fn, train=True):
    model.train(train)
    total_loss = 0
    all_preds, all_labels = [], []
    for xb, yb in tqdm(loader, leave=False):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        out = model(xb)
        loss = loss_fn(out, yb)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_loss += loss.item()
        preds = (torch.sigmoid(out) > 0.5).float().detach().cpu().numpy().ravel()
        labels = yb.detach().cpu().numpy().ravel()
        all_preds.extend(preds)
        all_labels.extend(labels)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    return total_loss / len(loader), acc, prec, rec, f1

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    print(f"[INFO] Using device: {DEVICE}")
    tr = make_loader(LANG, "train", BATCH, True)
    va = make_loader(LANG, "valid", BATCH, False)
    model = BiLSTM().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    for e in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_p, tr_r, tr_f1 = run_epoch(model, tr, opt, loss_fn, True)
        va_loss, va_acc, va_p, va_r, va_f1 = run_epoch(model, va, opt, loss_fn, False)
        print(f"Epoch {e}/{EPOCHS} | Train loss={tr_loss:.4f}, F1={tr_f1:.4f} | Val loss={va_loss:.4f}, F1={va_f1:.4f}")

        torch.save(model.state_dict(), os.path.join(OUT_DIR, f"lstm_unc_{LANG}_e{e}.pt"))

    final_path = os.path.join(OUT_DIR, f"lstm_uncentered_{LANG}.pt")
    torch.save(model.state_dict(), final_path)
    print(f"[DONE] Saved final model → {final_path}")

    # Metadata
    meta = {
        "language": LANG,
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "learning_rate": LR,
        "embedding_dim": 32,
        "hidden_dim": 128,
        "device": DEVICE,
        "architecture": "BiLSTM (Uncentered)",
        "window_chars": 100
    }
    with open(os.path.join(OUT_DIR, f"lstm_uncentered_{LANG}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
