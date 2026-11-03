"""
models.py - the three architectures we are comparing.

All three take the SAME input and produce the SAME kind of output:
    input   noisy magnitude spectrogram          (batch, 257 freq bins, 376 frames)
    output  a mask in [0, 1] of the same shape    -> denoised = mask * noisy

What differs is only HOW each network looks at the spectrogram:
    DNN   one frame at a time (plus a few neighbouring frames glued on), no memory
    CNN   the whole spectrogram as an image: small filters over time x frequency patches
    LSTM  frame by frame left-to-right, carrying a memory of what came before

Every model gets the input as log1p(magnitude), standardised with the training-set
mean/std (stored inside the model so a checkpoint is self-contained).
"""
import torch
from torch import nn
import torch.nn.functional as F

N_FREQ, N_FRAMES = 257, 376


class MaskModel(nn.Module):
    """Shared wrapper: input normalisation in, mask (with optional floor) out."""

    def __init__(self, mask_floor=0.0):
        super().__init__()
        self.mask_floor = mask_floor
        self.register_buffer("in_mean", torch.tensor(0.0))
        self.register_buffer("in_std", torch.tensor(1.0))

    def set_norm(self, mean, std):
        self.in_mean.fill_(float(mean))
        self.in_std.fill_(float(std))

    def normalise(self, mag):
        return (torch.log1p(mag) - self.in_mean) / self.in_std

    def to_mask(self, logits):
        return self.mask_floor + (1.0 - self.mask_floor) * torch.sigmoid(logits)

    def forward(self, mag):                       # mag: (B, 257, T) linear magnitude
        return self.to_mask(self.net_forward(self.normalise(mag)))


class DNNDenoiser(MaskModel):
    """Fully connected network. Each output frame sees a window of `context` frames
    on each side (like the MathWorks 'Denoise Speech Using Deep Learning' example),
    but has no memory and no notion of which frequency bins are neighbours."""

    def __init__(self, hidden=384, layers=3, context=4, dropout=0.2, mask_floor=0.0):
        super().__init__(mask_floor)
        self.context = context
        dims = [N_FREQ * (2 * context + 1)] + [hidden] * (layers - 1)
        blocks = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks += [nn.Linear(a, b), nn.LayerNorm(b), nn.ReLU(), nn.Dropout(dropout)]
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(dims[-1], N_FREQ)

    def net_forward(self, x):                     # x: (B, 257, T)
        B, Fq, T = x.shape
        k = 2 * self.context + 1
        x = F.pad(x, (self.context, self.context))            # pad time
        win = x.unfold(2, k, 1)                                # (B, 257, T, k)
        win = win.permute(0, 2, 1, 3).reshape(B, T, Fq * k)    # (B, T, 257*k)
        return self.head(self.body(win)).transpose(1, 2)      # (B, 257, T)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class CNNDenoiser(MaskModel):
    """2D U-Net over the spectrogram image (same architecture as Yuki's
    src/cnn/model.py on the unet-train-eval branch, so our CNN result lines up with
    the team's CNN work). Encoder shrinks the image 3x, decoder grows it back, and
    skip connections carry fine detail across."""

    PAD_F, PAD_T = 264, 384                       # divisible by 8 for three 2x poolings

    def __init__(self, base_channels=16, dropout=0.0, mask_floor=0.0):
        super().__init__(mask_floor)
        c = base_channels
        self.enc1, self.enc2, self.enc3 = ConvBlock(1, c), ConvBlock(c, 2 * c), ConvBlock(2 * c, 4 * c)
        self.bottleneck = ConvBlock(4 * c, 4 * c)
        self.drop = nn.Dropout2d(dropout)
        self.dec3 = ConvBlock(8 * c, 4 * c)
        self.dec2 = ConvBlock(6 * c, 2 * c)
        self.dec1 = ConvBlock(3 * c, c)
        self.out = nn.Conv2d(c, 1, 1)

    @staticmethod
    def _up(h, like):
        return F.interpolate(h, size=like.shape[-2:], mode="bilinear", align_corners=False)

    def net_forward(self, x):                     # x: (B, 257, T)
        Fq, T = x.shape[-2:]
        x = F.pad(x.unsqueeze(1), (0, self.PAD_T - T, 0, self.PAD_F - Fq))
        s1 = self.enc1(x)
        s2 = self.enc2(F.max_pool2d(s1, 2))
        s3 = self.enc3(F.max_pool2d(s2, 2))
        h = self.drop(self.bottleneck(F.max_pool2d(s3, 2)))
        h = self.dec3(torch.cat([self._up(h, s3), s3], 1))
        h = self.dec2(torch.cat([self._up(h, s2), s2], 1))
        h = self.dec1(torch.cat([self._up(h, s1), s1], 1))
        return self.out(h)[:, 0, :Fq, :T]


class LSTMDenoiser(MaskModel):
    """Unidirectional LSTM (same shape as Sophia's src/train_lstm.py on lstm-model):
    reads one 257-bin frame per time step and keeps a memory. Unidirectional means
    it never looks ahead, so it could run in real time (our stretch goal)."""

    def __init__(self, hidden=256, layers=2, dropout=0.2, mask_floor=0.0):
        super().__init__(mask_floor)
        self.lstm = nn.LSTM(N_FREQ, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, N_FREQ)

    def net_forward(self, x):                     # x: (B, 257, T)
        h, _ = self.lstm(x.transpose(1, 2))       # (B, T, hidden)
        return self.head(h).transpose(1, 2)


class CRNDenoiser(MaskModel):
    """CNN + LSTM together (a Convolutional Recurrent Network, Tan & Wang 2018).

    Encoder: 5 conv layers, each halving FREQUENCY only (257->129->65->33->17->9) so every
             time frame is kept - the CNN part learns local time-frequency patterns.
    Middle:  an LSTM reads the encoded frames left to right - the memory part.
    Decoder: transposed convs grow frequency back to 257, with U-Net style skip connections.
    """

    def __init__(self, channels=(16, 32, 32, 64, 64), hidden=256, layers=2, dropout=0.2, mask_floor=0.0):
        super().__init__(mask_floor)
        chs = [1, *channels]
        self.enc = nn.ModuleList([
            nn.Sequential(nn.Conv2d(chs[i], chs[i + 1], 3, stride=(2, 1), padding=1),
                          nn.BatchNorm2d(chs[i + 1]), nn.ELU())
            for i in range(len(channels))])
        f = N_FREQ
        for _ in channels:
            f = (f + 1) // 2                                   # 257 -> ... -> 9
        self.f_bottom = f
        feat = channels[-1] * f
        self.lstm = nn.LSTM(feat, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.proj = nn.Linear(hidden, feat)
        dec = []
        for i in range(len(channels), 0, -1):                  # mirror of the encoder
            cin, cout = chs[i] * 2, chs[i - 1] if i > 1 else 8
            dec.append(nn.Sequential(nn.ConvTranspose2d(cin, cout, 3, stride=(2, 1), padding=1),
                                     nn.BatchNorm2d(cout), nn.ELU()))
        self.dec = nn.ModuleList(dec)
        self.out = nn.Conv2d(8, 1, 1)

    def net_forward(self, x):                     # x: (B, 257, T)
        h = x.unsqueeze(1)                        # (B, 1, 257, T)
        skips = []
        for layer in self.enc:
            h = layer(h)
            skips.append(h)
        B, C, Fb, T = h.shape
        z, _ = self.lstm(h.permute(0, 3, 1, 2).reshape(B, T, C * Fb))
        h = self.proj(z).reshape(B, T, C, Fb).permute(0, 2, 3, 1)
        for layer, skip in zip(self.dec, reversed(skips)):
            h = layer(torch.cat([h, skip], 1))
        return self.out(h)[:, 0]                  # (B, 257, T)


MODELS = {"dnn": DNNDenoiser, "cnn": CNNDenoiser, "lstm": LSTMDenoiser, "crn": CRNDenoiser}


def build_model(name, mask_floor=0.0, **arch):
    return MODELS[name](mask_floor=mask_floor, **arch)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
