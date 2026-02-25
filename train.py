import json
import os

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import KhmerOCRDataset, collate_fn
from model import KhmerOCR

HIDDEN_DIM = 256
EMB_DIM = 128
BATCH_SIZE = 64
NUM_EPOCHS = 100
LR = 1e-3
IMG_HEIGHT = 32
# Teacher forcing starts at 1.0 (full teacher forcing) and decays to 0.0.
# This helps early convergence and gradually shifts the model to rely on its own predictions.
TF_START = 1.0
TF_END = 0.0


def train_epoch(model, loader, optimizer, criterion, device, teacher_forcing_ratio):
  model.train()
  total_loss = 0.0
  bar = tqdm(loader, desc="Train")
  for imgs, labels in bar:
    imgs = imgs.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    outputs = model(imgs, labels, teacher_forcing_ratio=teacher_forcing_ratio)
    B, T, V = outputs.shape
    loss = criterion(outputs.reshape(B * T, V), labels[:, 1:].reshape(B * T))
    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    total_loss += loss.item()
    bar.set_postfix(loss=f"{loss.item():.4f}")
  return total_loss / len(loader)


def eval_epoch(model, loader, criterion, device):
  model.eval()
  total_loss = 0.0
  correct = 0
  total = 0
  bar = tqdm(loader, desc="Eval")
  with torch.no_grad():
    for imgs, labels in bar:
      imgs = imgs.to(device)
      labels = labels.to(device)
      outputs = model(imgs, labels, teacher_forcing_ratio=0.0)
      B, T, V = outputs.shape
      loss = criterion(outputs.reshape(B * T, V), labels[:, 1:].reshape(B * T))
      total_loss += loss.item()
      preds = outputs.argmax(dim=-1)
      mask = labels[:, 1:] != 0
      correct += (preds == labels[:, 1:]).masked_select(mask).sum().item()
      total += mask.sum().item()
      acc = correct / total if total > 0 else 0.0
      bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")
  return total_loss / len(loader), correct / total if total > 0 else 0.0


def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  root = os.path.dirname(os.path.abspath(__file__))

  train_ds = KhmerOCRDataset(
    os.path.join(root, "dataset/train.tsv"), root, img_height=IMG_HEIGHT
  )
  test_ds = KhmerOCRDataset(
    os.path.join(root, "dataset/test.tsv"),
    root,
    vocab=train_ds.vocab,
    img_height=IMG_HEIGHT,
  )

  train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
  )
  test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
  )

  model = KhmerOCR(
    vocab_size=len(train_ds.vocab),
    hidden_dim=HIDDEN_DIM,
    device=device,
    emb_dim=EMB_DIM,
  ).to(device)

  warmup_epochs = 5
  optimizer = torch.optim.Adam(model.parameters(), lr=LR)
  warmup_sch = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1,
    total_iters=warmup_epochs,
  )
  cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=(NUM_EPOCHS - warmup_epochs)
  )
  scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_sch, cosine_sch], milestones=[warmup_epochs]
  )

  criterion = nn.CrossEntropyLoss(ignore_index=0)

  # Save vocab separately so it doesn't bloat the model checkpoint
  vocab_path = os.path.join(root, "vocab.json")
  with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(train_ds.vocab.char2idx, f, ensure_ascii=False, indent=2)

  best_loss = float("inf")
  for epoch in range(NUM_EPOCHS):
    # Linear decay: 1.0 at epoch 0, 0.0 at final epoch
    teacher_forcing_ratio = TF_START - (TF_START - TF_END) * (epoch / (NUM_EPOCHS - 1))
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device, teacher_forcing_ratio)
    val_loss, val_acc = eval_epoch(model, test_loader, criterion, device)
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(
      f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
      f"Train Loss: {train_loss:.4f} | "
      f"Val Loss: {val_loss:.4f} | "
      f"Val Acc: {val_acc:.4f} | "
      f"LR: {current_lr:.6f} | "
      f"TF: {teacher_forcing_ratio:.2f}"
    )
    if val_loss < best_loss:
      best_loss = val_loss
      torch.save(model.state_dict(), os.path.join(root, "best.pt"))
      print(f"  -> Saved best model (val_loss={val_loss:.4f})")


if __name__ == "__main__":
  main()
