import argparse
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from data import Vocabulary
from model import KhmerOCR

HIDDEN_DIM = 256
EMB_DIM = 128
IMG_HEIGHT = 32
MAX_DECODE_LEN = 512

SOS = 1
EOS = 2


def load_vocab(vocab_path):
    with open(vocab_path, encoding="utf-8") as f:
        char2idx = json.load(f)
    vocab = Vocabulary()
    vocab.char2idx = char2idx
    vocab.idx2char = {v: k for k, v in char2idx.items()}
    return vocab


def load_model(checkpoint_path, vocab_path, device):
    vocab = load_vocab(vocab_path)
    model = KhmerOCR(
        vocab_size=len(vocab),
        hidden_dim=HIDDEN_DIM,
        emb_dim=EMB_DIM,
        device=device,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, vocab


def preprocess(image_path, img_height=IMG_HEIGHT):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    new_w = max(1, int(w * img_height / h))
    img = img.resize((new_w, img_height), Image.BILINEAR)
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    return transform(img).unsqueeze(0)  # [1, 1, H, W]


@torch.no_grad()
def decode(model, img_tensor, vocab, device, max_len=MAX_DECODE_LEN):
    img_tensor = img_tensor.to(device)
    enc_outputs, hidden = model.encoder(img_tensor)

    input_token = torch.tensor([SOS], dtype=torch.long, device=device)
    result = []

    for _ in range(max_len):
        output, hidden = model.decoder(input_token, hidden, enc_outputs)
        top1 = output.argmax(1)
        token_id = top1.item()
        if token_id == EOS:
            break
        result.append(token_id)
        input_token = top1

    return vocab.decode(result)


def main():
    parser = argparse.ArgumentParser(description="Khmer OCR inference")
    parser.add_argument("images", nargs="+", help="Image file path(s) to recognize")
    parser.add_argument(
        "--checkpoint",
        default="best.pt",
        help="Path to model checkpoint (default: best.pt)",
    )
    parser.add_argument(
        "--vocab",
        default="vocab.json",
        help="Path to vocab.json (default: vocab.json)",
    )
    parser.add_argument(
        "--img-height",
        type=int,
        default=IMG_HEIGHT,
        help=f"Image height for resizing (default: {IMG_HEIGHT})",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=MAX_DECODE_LEN,
        help=f"Max decode length (default: {MAX_DECODE_LEN})",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    if not os.path.isfile(args.checkpoint):
        print(f"Error: checkpoint not found: {args.checkpoint}")
        raise SystemExit(1)

    if not os.path.isfile(args.vocab):
        print(f"Error: vocab not found: {args.vocab}")
        raise SystemExit(1)

    model, vocab = load_model(args.checkpoint, args.vocab, device)

    for image_path in args.images:
        if not os.path.isfile(image_path):
            print(f"{image_path}\t[error: file not found]")
            continue
        img_tensor = preprocess(image_path, img_height=args.img_height)
        text = decode(model, img_tensor, vocab, device, max_len=args.max_len)
        print(f"{image_path}\t{text}")


if __name__ == "__main__":
    main()
