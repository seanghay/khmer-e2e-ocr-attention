import csv
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

PAD = 0
SOS = 1
EOS = 2


class Vocabulary:
  def __init__(self):
    self.char2idx = {"<PAD>": PAD, "<SOS>": SOS, "<EOS>": EOS}
    self.idx2char = {PAD: "<PAD>", SOS: "<SOS>", EOS: "<EOS>"}

  def build(self, texts):
    chars = sorted(set(ch for text in texts for ch in text))
    for ch in chars:
      if ch not in self.char2idx:
        idx = len(self.char2idx)
        self.char2idx[ch] = idx
        self.idx2char[idx] = ch

  def encode(self, text):
    return [self.char2idx[ch] for ch in text if ch in self.char2idx]

  def decode(self, indices):
    return "".join(
      self.idx2char[i]
      for i in indices
      if i not in (PAD, SOS, EOS) and i in self.idx2char
    )

  def __len__(self):
    return len(self.char2idx)


class KhmerOCRDataset(Dataset):
  def __init__(self, tsv_path, root_dir, vocab=None, img_height=32):
    self.root_dir = root_dir
    self.img_height = img_height
    self.samples = []

    with open(tsv_path, newline="", encoding="utf-8") as f:
      reader = csv.reader(f, delimiter="\t")
      for row in reader:
        if len(row) == 2:
          self.samples.append((row[0], row[1]))

    if vocab is None:
      self.vocab = Vocabulary()
      self.vocab.build([text for _, text in self.samples])
    else:
      self.vocab = vocab

    self.transform = transforms.Compose([
      transforms.Grayscale(),
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,)),
    ])

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    img_path, text = self.samples[idx]
    full_path = os.path.join(self.root_dir, img_path)
    img = Image.open(full_path).convert("RGB")

    w, h = img.size
    new_w = max(1, int(w * self.img_height / h))
    img = img.resize((new_w, self.img_height), Image.BILINEAR)

    img_tensor = self.transform(img)
    tokens = [SOS] + self.vocab.encode(text) + [EOS]
    token_tensor = torch.tensor(tokens, dtype=torch.long)
    return img_tensor, token_tensor


def collate_fn(batch):
  imgs, labels = zip(*batch)
  max_w = max(img.shape[2] for img in imgs)
  padded_imgs = torch.stack([
    F.pad(img, (0, max_w - img.shape[2])) for img in imgs
  ])
  max_len = max(lbl.shape[0] for lbl in labels)
  padded_labels = torch.stack([
    F.pad(lbl, (0, max_len - lbl.shape[0]), value=PAD) for lbl in labels
  ])
  return padded_imgs, padded_labels


if __name__ == "__main__":
  from torch.utils.data import DataLoader

  root = os.path.dirname(os.path.abspath(__file__))
  train_ds = KhmerOCRDataset(os.path.join(root, "dataset/train.tsv"), root)
  test_ds = KhmerOCRDataset(
    os.path.join(root, "dataset/test.tsv"), root, vocab=train_ds.vocab
  )

  print(f"Vocab size: {len(train_ds.vocab)}")
  print(f"Train samples: {len(train_ds)}")
  print(f"Test samples: {len(test_ds)}")

  loader = DataLoader(train_ds, batch_size=4, collate_fn=collate_fn)
  imgs, labels = next(iter(loader))
  print(f"Image batch shape: {imgs.shape}")
  print(f"Label batch shape: {labels.shape}")
