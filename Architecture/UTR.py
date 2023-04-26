import torch
import torch.nn as nn


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x.unsqueeze(2) ** 2, dim=1) + self.eps)
        x = x / norm
        return x


def input_norm(x):
    data_mean = torch.mean(x, dim=[2, 3]).unsqueeze(-1).unsqueeze(-1)
    data_std = torch.std(x, dim=[2, 3]).unsqueeze(-1).unsqueeze(-1) + 1e-10
    data_norm = (x - data_mean.detach()) / data_std.detach()
    return data_norm


class conv(nn.Module):
    def __init__(self, inch, outch, kernal_size=3, stride=1, padding=0):
        super(conv, self).__init__()
        if padding is None:
            padding = (kernal_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv2d(inch, outch, kernal_size, stride, padding, bias=False),
            nn.BatchNorm2d(outch, affine=False),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class UTRNet(nn.Module):
    def __init__(self, model_path=None, pad=None):
        super(UTRNet, self).__init__()
        if pad is not None:
            first_conv = conv(1, 32, padding=pad)
        else:
            first_conv = conv(1, 32, )
        self.net = nn.Sequential(
            first_conv,
            nn.MaxPool2d(2),
            conv(32, 32),
            conv(32, 64),
            nn.MaxPool2d(2),
            conv(64, 64),
            conv(64, 128),
            conv(128, 128),
            conv(128, 256),
            conv(256, 128),
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128, affine=False)
        )
        if model_path is not None:
            self.load_state_dict(torch.load(model_path))

    def forward(self, x):
        x_t = input_norm(x)
        x_t = self.net(x_t)
        x = L2Norm()(x_t)
        return x
