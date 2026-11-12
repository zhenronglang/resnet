"""Configurable residual networks for 32x32 CIFAR-10 images.

Every model here is a ResNet in the sense of the assignment: a stack of residual
blocks computing  S(x) + F(x)  where S is the skip connection (identity, or a
K x K conv when the shape changes) and F is a small conv/BN/ReLU stack.

Design variables exposed (names follow the assignment PDF):
    widths   C_i : channels of each stage
    depths   N_i : number of residual blocks per stage
    kernel   F_i : conv kernel size inside the blocks
    skip_k   K_i : kernel size of the projection skip connections
    pool     P   : output size of the final adaptive average pool
    upsample     : optional input resolution; the 32x32 image is bilinearly resized to it inside forward()
                   (no parameters, more compute per parameter)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Original (post-activation) block, as in the assignment:
    out = ReLU(S(x) + F(x)),   F = conv-BN-ReLU-conv-BN."""

    def __init__(self, cin, cout, stride, k=3, skip_k=1, drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, k, stride, k // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, k, 1, k // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.drop = drop
        self.shortcut = nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cin, cout, skip_k, stride, skip_k // 2, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.drop > 0:
            out = F.dropout(out, self.drop, self.training)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class PreActBlock(nn.Module):
    """Pre-activation block (He et al. 2016, "Identity Mappings"), used by Wide ResNets:
    out = S(x) + F(x),   F = BN-ReLU-conv-BN-ReLU-conv."""

    def __init__(self, cin, cout, stride, k=3, skip_k=1, drop=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, k, stride, k // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, k, 1, k // 2, bias=False)
        self.drop = drop
        self.shortcut = None
        if stride != 1 or cin != cout:
            self.shortcut = nn.Conv2d(cin, cout, skip_k, stride, skip_k // 2, bias=False)

    def forward(self, x):
        o = F.relu(self.bn1(x))
        sc = x if self.shortcut is None else self.shortcut(o)
        o = self.conv1(o)
        o = F.relu(self.bn2(o))
        if self.drop > 0:
            o = F.dropout(o, self.drop, self.training)
        return self.conv2(o) + sc


class ResNet(nn.Module):
    def __init__(self, widths=(64, 128, 256), depths=(3, 3, 3), block="preact", stem=16,
                 kernel=3, skip_k=1, pool=1, drop=0.0, upsample=None, num_classes=10):
        super().__init__()
        self.upsample = upsample
        Block = {"preact": PreActBlock, "basic": BasicBlock}[block]
        self.preact = block == "preact"
        self.stem = nn.Conv2d(3, stem, 3, 1, 1, bias=False)
        self.stem_bn = None if self.preact else nn.BatchNorm2d(stem)
        layers, cin = [], stem
        for i, (w, n) in enumerate(zip(widths, depths)):
            for j in range(n):
                stride = 2 if (i > 0 and j == 0) else 1
                layers.append(Block(cin, w, stride, kernel, skip_k, drop))
                cin = w
        self.layers = nn.Sequential(*layers)
        self.final_bn = nn.BatchNorm2d(cin) if self.preact else None
        self.pool = nn.AdaptiveAvgPool2d(pool)
        self.fc = nn.Linear(cin * pool * pool, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.upsample and x.shape[-1] != self.upsample:
            x = F.interpolate(x, size=(self.upsample, self.upsample), mode="bilinear", align_corners=False)
        x = self.stem(x)
        if self.stem_bn is not None:
            x = F.relu(self.stem_bn(x))
        x = self.layers(x)
        if self.final_bn is not None:
            x = F.relu(self.final_bn(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def shake_drop(x, p_survive, training):
    """ShakeDrop (Yamada et al. 2019), per sample. Forward: x * (b + alpha - b*alpha), backward: grad * (b + beta -
    b*beta), b ~ Bernoulli(p_survive), alpha ~ U[-1, 1], beta ~ U[0, 1]. At test time: x * E[...] = x * p_survive."""
    if not training:
        return x * p_survive
    shape = (x.shape[0], 1, 1, 1)
    b = torch.bernoulli(torch.full(shape, p_survive, device=x.device, dtype=x.dtype))
    alpha = torch.empty(shape, device=x.device, dtype=x.dtype).uniform_(-1, 1)
    beta = torch.rand(shape, device=x.device, dtype=x.dtype)
    m_fwd, m_bwd = b + alpha - b * alpha, b + beta - b * beta
    return m_bwd * x + ((m_fwd - m_bwd) * x).detach()


class PyramidBlock(nn.Module):
    """PyramidNet basic block (Han et al. 2017): BN-conv-BN-ReLU-conv-BN, no ReLU after the addition; the skip
    connection is identity (avg-pooled when strided) with zero-padded extra channels."""

    def __init__(self, cin, cout, stride, p_survive):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(cout)
        self.stride, self.pad, self.p = stride, cout - cin, p_survive

    def forward(self, x):
        o = self.bn3(self.conv2(F.relu(self.bn2(self.conv1(self.bn1(x))))))
        o = shake_drop(o, self.p, self.training)
        sc = F.avg_pool2d(x, 2) if self.stride == 2 else x
        if self.pad:
            sc = F.pad(sc, (0, 0, 0, 0, 0, self.pad))
        return o + sc


class PyramidNet(nn.Module):
    """PyramidNet with ShakeDrop (basic blocks). depth = 6N + 2; channel width grows linearly by alpha in total."""

    def __init__(self, depth=110, alpha=270, p_last=0.5, num_classes=10):
        super().__init__()
        n = (depth - 2) // 6
        L = 3 * n
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16))
        layers, w = [], 16.0
        for i in range(L):
            cin = int(round(w))
            w += alpha / L
            stride = 2 if i in (n, 2 * n) else 1
            layers.append(PyramidBlock(cin, int(round(w)), stride, 1 - p_last * (i + 1) / L))
        self.layers = nn.Sequential(*layers)
        cout = int(round(w))
        self.final_bn = nn.BatchNorm2d(cout)
        self.fc = nn.Linear(cout, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.layers(self.stem(x))
        x = F.relu(self.final_bn(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


# Named configurations. Students must stay <= 5M trainable parameters; teachers
# (only used for knowledge distillation during training) are unconstrained.
CONFIGS = {
    # --- student candidates (<= 5M params) ---
    "wrn16_4":     dict(widths=(64, 128, 256), depths=(2, 2, 2)),
    "wrn22_4":     dict(widths=(64, 128, 256), depths=(3, 3, 3)),
    "res443":      dict(widths=(64, 128, 256), depths=(4, 4, 3)),
    "res443_post": dict(widths=(64, 128, 256), depths=(4, 4, 3), block="basic", stem=64),
    "post643":     dict(widths=(64, 128, 256), depths=(6, 4, 3), block="basic", stem=64),
    "post_w80":    dict(widths=(80, 160, 256), depths=(3, 3, 3), block="basic", stem=64),
    "pre_w80":     dict(widths=(80, 160, 256), depths=(3, 3, 3)),
    "rn18_w40":    dict(widths=(40, 80, 160, 320), depths=(2, 2, 2, 2), block="basic", stem=40),
    # --- teachers ---
    "wrn28_10":    dict(widths=(160, 320, 640), depths=(4, 4, 4), drop=0.0),
    "wrn40_4":     dict(widths=(64, 128, 256), depths=(6, 6, 6)),
    "pyramid110_270": dict(net="pyramid", depth=110, alpha=270),
    "pyramid164_270": dict(net="pyramid", depth=164, alpha=270),
}


def build_model(name, **overrides):
    cfg = dict(CONFIGS[name])
    cfg.update(overrides)
    if cfg.pop("net", "resnet") == "pyramid":
        cfg.pop("drop", None)
        return PyramidNet(**cfg)
    return ResNet(**cfg)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name in CONFIGS:
        m = build_model(name)
        x = torch.randn(2, 3, 32, 32)
        assert m(x).shape == (2, 10)
        print(f"{name:12s} {count_params(m):>12,d} params")
