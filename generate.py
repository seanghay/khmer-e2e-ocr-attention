import csv
import os
import random
import re
from tqdm import tqdm
from multiprocessing import Pool
from normalize import khnormal
from glob import glob
from PIL import Image, ImageFont, ImageDraw


def get_fonts():
  return glob("fonts/**/*.ttf", recursive=True)


def create_text_line(values):
  p, text, font_file = values
  font = ImageFont.truetype(font_file, 64)
  left, top, right, bottom = font.getbbox(text)

  p_top = random.randint(0, 50)
  p_left = random.randint(0, 50)
  p_right = random.randint(0, 50)
  p_bottom = random.randint(0, 50)

  w = right - left + p_left + p_right
  h = bottom - top + p_top + p_bottom

  c = random.randint(120, 255)
  im = Image.new("RGB", (w, h), (c, c, c))

  draw = ImageDraw.Draw(im)
  c = random.randint(0, 100)
  draw.text((-left + p_left, -top + p_top), text, fill=(c, c, c), font=font)
  im = im.convert("L")
  im = im.resize((int(32 * (im.width / im.height)), 32))
  im.save(p, format="JPEG", quality=random.randint(50, 90))


if __name__ == "__main__":
  with open("data.txt") as infile:
    text = infile.read().replace("\n", " ")
  text = text.replace("\u200b", "")
  max_length = 128
  lines = []
  fonts = get_fonts()
  i = 0

  os.makedirs(os.path.join("dataset", "train"), exist_ok=True)
  os.makedirs(os.path.join("dataset", "test"), exist_ok=True)

  train_set = []
  test_set = []

  for m in re.finditer(r"([\u1780-\u17ff\.]+)", text):
    chunk = m[0]
    if len(chunk) > max_length:
      continue
    chunk = khnormal(chunk)
    subset = "train" if i > 100 else "test"
    p = os.path.join("dataset", subset, "img_" + str(i).zfill(6) + ".jpg")
    lines.append((p, chunk, random.choice(fonts)))
    i += 1

    if subset == "train":
      train_set.append([p, chunk])
    else:
      test_set.append([p, chunk])

  with open(os.path.join("dataset", "train.tsv"), "w") as outfile:
    csv.writer(outfile, delimiter="\t").writerows(train_set)

  with open(os.path.join("dataset", "test.tsv"), "w") as outfile:
    csv.writer(outfile, delimiter="\t").writerows(test_set)

  with tqdm(total=len(lines)) as pbar:
    with Pool() as pool:
      for result in pool.imap_unordered(create_text_line, lines):
        pbar.update()
