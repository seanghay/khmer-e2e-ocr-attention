import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1):
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(
      in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(
      out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
    )

    self.bn2 = nn.BatchNorm2d(out_channels)
    self.shortcut = nn.Sequential()
    if stride != 1 or in_channels != out_channels:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels),
      )

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    return F.relu(out)


class Encoder(nn.Module):
  def __init__(self, hidden_dim):
    super(Encoder, self).__init__()
    self.layer1 = ResidualBlock(1, 64)
    self.layer2 = ResidualBlock(64, 128, stride=2)
    self.layer3 = ResidualBlock(128, 256, stride=2)
    self.layer4 = ResidualBlock(256, 512, stride=2)
    self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
    self.rnn = nn.GRU(512, hidden_dim, batch_first=True, bidirectional=True)
    self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

  def forward(self, x):
    # x: [Batch, 1, H, W]
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.adaptive_pool(x)  # [B, 512, 1, W_seq]
    x = x.squeeze(2).permute(0, 2, 1)  # [B, W_seq, 512]

    outputs, hidden = self.rnn(x)
    # outputs: [B, W_seq, hidden*2]
    # hidden: [2, B, hidden] -> we combine directions for the initial decoder state
    hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

    return outputs, hidden


class Attention(nn.Module):
  def __init__(self, enc_hid_dim, dec_hid_dim):
    super(Attention, self).__init__()
    self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
    self.v = nn.Linear(dec_hid_dim, 1, bias=False)

  def forward(self, hidden, encoder_outputs):
    src_len = encoder_outputs.shape[1]
    hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
    energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
    attention = self.v(energy).squeeze(2)
    return F.softmax(attention, dim=1)


class Decoder(nn.Module):
  def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim):
    super(Decoder, self).__init__()
    self.embedding = nn.Embedding(output_dim, emb_dim)
    self.attention = Attention(enc_hid_dim, dec_hid_dim)
    self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim, batch_first=True)
    self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

  def forward(self, input, hidden, encoder_outputs):
    # input: [B] (previous predicted character)
    input = input.unsqueeze(1)
    embedded = self.embedding(input)  # [B, 1, emb_dim]

    # Calculate attention weights
    a = self.attention(hidden, encoder_outputs).unsqueeze(1)  # [B, 1, seq_len]

    # Weighted sum of encoder outputs (context vector)
    weighted = torch.bmm(a, encoder_outputs)  # [B, 1, enc_hid_dim * 2]

    rnn_input = torch.cat((embedded, weighted), dim=2)
    output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

    # Output prediction
    prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=2))
    return prediction.squeeze(1), hidden.squeeze(0)


class KhmerOCR(nn.Module):
  def __init__(
    self,
    vocab_size,
    hidden_dim,
    device,
    emb_dim=128,
  ):
    super(KhmerOCR, self).__init__()
    self.encoder = Encoder(hidden_dim=hidden_dim)
    self.decoder = Decoder(
      output_dim=vocab_size,
      emb_dim=emb_dim,
      enc_hid_dim=hidden_dim,
      dec_hid_dim=hidden_dim,
    )
    self.device = device

  def forward(self, src_img, trg, teacher_forcing_ratio=0.5):
    batch_size = src_img.shape[0]
    trg_len = trg.shape[1]
    vocab_size = self.decoder.fc_out.out_features
    outputs = torch.zeros(trg_len - 1, batch_size, vocab_size).to(src_img.device)
    enc_outputs, hidden = self.encoder(src_img)
    input = trg[:, 0]

    for t in range(trg_len - 1):
      output, hidden = self.decoder(input, hidden, enc_outputs)
      outputs[t] = output
      top1 = output.argmax(1)
      use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
      input = trg[:, t + 1] if use_teacher_forcing else top1
    return outputs.permute(1, 0, 2)


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  HIDDEN_DIM = 256
  VOCAB_SIZE = 120
  model = KhmerOCR(hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE, device=device).to(
    device
  )
  dummy_img = torch.randn(1, 1, 64, 512).to(device)
  dummy_trg = torch.randint(0, VOCAB_SIZE, (1, 50)).to(device)
  output = model(dummy_img, dummy_trg, teacher_forcing_ratio=0.5)
  print(output.shape)
