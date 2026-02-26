import csv
import os
import random

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

PAD = 0
SOS = 1
EOS = 2

# Augmentation probability split (training only)
BLANK_PROB    = 0.05   # blank/noise image → empty label (anti-hallucination)
PHYSICAL_PROB = 0.425  # physical document pipeline
DIGITAL_PROB  = 0.425  # digital document pipeline
# remaining ~10% → clean pass-through


def build_physical_aug():
  """Augmentation pipeline simulating scanned or photographed physical documents."""
  return A.Compose([
    A.OneOf([
      # std_range is a fraction of 255; (0.04, 0.20) ≈ 10–51 px std
      A.GaussNoise(std_range=(0.04, 0.20), p=1.0),
      A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
    ], p=0.5),
    A.OneOf([
      A.MotionBlur(blur_limit=3, p=1.0),
      A.GaussianBlur(blur_limit=3, p=1.0),
    ], p=0.4),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
    A.OneOf([
      A.OpticalDistortion(distort_limit=0.03, p=1.0),
      A.GridDistortion(num_steps=3, distort_limit=0.05, p=1.0),
    ], p=0.3),
    A.ImageCompression(quality_range=(50, 85), p=0.5),
    A.Rotate(limit=2, border_mode=4, p=0.4),  # border_mode=4 → BORDER_REFLECT_101
    A.CoarseDropout(
      num_holes_range=(1, 4),
      hole_height_range=(1, 4),
      hole_width_range=(2, 8),
      fill=255, p=0.3,
    ),
    A.RandomShadow(
      shadow_roi=(0, 0, 1, 1),
      num_shadows_limit=(1, 2),
      shadow_dimension=4, p=0.2,
    ),
  ])


def build_digital_aug():
  """Augmentation pipeline simulating screen-rendered or digital PDF documents."""
  return A.Compose([
    # std_range as fraction of 255; (0.01, 0.08) ≈ 2.5–20 px std
    A.GaussNoise(std_range=(0.01, 0.08), p=0.3),
    A.OneOf([
      A.Sharpen(alpha=(0.1, 0.4), lightness=(0.9, 1.1), p=1.0),
      A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=1.0),
      A.UnsharpMask(blur_limit=3, p=1.0),
    ], p=0.4),
    A.HueSaturationValue(
      hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=15, p=0.4
    ),
    A.ImageCompression(quality_range=(75, 98), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.4),
    A.CoarseDropout(
      num_holes_range=(1, 2),
      hole_height_range=(1, 4),
      hole_width_range=(2, 6),
      fill=255, p=0.2,
    ),
  ])


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
  def __init__(self, tsv_path, root_dir, vocab=None, img_height=32, augment=False):
    self.root_dir = root_dir
    self.img_height = img_height
    self.augment = augment
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

    if augment:
      self.physical_aug = build_physical_aug()
      self.digital_aug = build_digital_aug()

  def _make_blank_tensor(self, width):
    """Return a tensor of a blank or heavily noised image with no text."""
    h, w = self.img_height, width
    r = random.random()
    if r < 0.5:
      # Solid white/light-gray background
      bg = random.randint(200, 255)
      img_np = np.full((h, w, 3), bg, dtype=np.uint8)
    elif r < 0.8:
      # Heavy random pixel noise
      img_np = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    else:
      # Gaussian noise around a gray mean
      mean = random.randint(150, 220)
      std = random.randint(30, 80)
      img_np = np.clip(
        np.random.normal(mean, std, (h, w, 3)), 0, 255
      ).astype(np.uint8)
    return self.transform(Image.fromarray(img_np))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    img_path, text = self.samples[idx]

    # Blank/noise sample — teaches the model to output nothing on empty inputs
    if self.augment and random.random() < BLANK_PROB:
      new_w = random.randint(32, 320)
      img_tensor = self._make_blank_tensor(new_w)
      return img_tensor, torch.tensor([SOS, EOS], dtype=torch.long)

    full_path = os.path.join(self.root_dir, img_path)
    img = Image.open(full_path).convert("RGB")

    w, h = img.size
    new_w = max(1, int(w * self.img_height / h))
    img = img.resize((new_w, self.img_height), Image.BILINEAR)

    if self.augment:
      img_np = np.array(img)
      r = random.random()
      if r < PHYSICAL_PROB:
        img_np = self.physical_aug(image=img_np)["image"]
      elif r < PHYSICAL_PROB + DIGITAL_PROB:
        img_np = self.digital_aug(image=img_np)["image"]
      # else: ~10% clean pass-through — no augmentation
      img = Image.fromarray(img_np)

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
  train_ds = KhmerOCRDataset(
    os.path.join(root, "dataset/train.tsv"), root, augment=True
  )
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
