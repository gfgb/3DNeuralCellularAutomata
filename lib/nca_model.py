import torch
import torch.nn as nn
import torch.nn.functional as F

class NCAModel3D(nn.Module):
    def __init__(self, n_channels, fire_rate, device, hidden_size=128):
        super().__init__()
        self.device = device
        self.n_channels = n_channels

        self.fc0 = nn.Linear(n_channels * 4, hidden_size)
        self.fc1 = nn.Linear(hidden_size, n_channels, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.fire_rate = fire_rate
        self.to(self.device)
        self.dx, self.dy, self.dz = [ item.to(self.device) for item in NCAModel3D.sobel_filter_3d() ]

    def load_pretrained(self, path):
        self.load_state_dict(torch.load(path))

    @staticmethod
    def sobel_filter_3d():
        dx = torch.outer(torch.tensor([1, 2, 1], dtype=torch.float32),
                         torch.tensor([-1, 0, 1], dtype=torch.float32))
        dx = dx.repeat([3, 1, 1]) * torch.tensor([1, 2, 1]).reshape(3, 1, 1)
        dx = dx / dx.abs().sum()
        dy = dx.transpose(1, 2)
        dz = dx.transpose(0, 2)
        return dx.permute(1, 2, 0), dy.permute(1, 2, 0), dz.permute(1, 2, 0)

    @staticmethod
    def alive(x):
        return F.max_pool3d(x[:, 3:4, :, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x):

        def _perceive_with(x_, conv_weights):
            conv_weights = conv_weights.view(1, 1, 3, 3, 3).repeat(self.n_channels, 1, 1, 1, 1)
            return F.conv3d(x_, conv_weights, padding=1, groups=self.n_channels)

        y1 = _perceive_with(x, self.dx)
        y2 = _perceive_with(x, self.dy)
        y3 = _perceive_with(x, self.dz)
        y = torch.cat((x, y1, y2, y3), dim=1)
        return y

    def update(self, x, fire_rate):

        x = x.transpose(1, 4)

        pre_life_mask = self.alive(x)

        dx = self.perceive(x)

        dx = dx.transpose(1, 4)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        fire_rate = self.fire_rate if fire_rate is None else fire_rate

        stochastic = torch.rand([dx.size(0),
                                 dx.size(1),
                                 dx.size(2),
                                 dx.size(3), 1], dtype=torch.float32) > fire_rate

        stochastic = stochastic.to(self.device)
        dx = dx * stochastic

        x = x + dx.transpose(1, 4)

        post_life_mask = self.alive(x)

        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask

        return x.transpose(1, 4)

    def forward(self, x, steps=1, fire_rate=None):
        for step in range(steps):
            x = self.update(x, fire_rate)
        return x

class NCAModel3D_conv(nn.Module):
    def __init__(self, n_channels, fire_rate, device, hidden_size=128):
        super().__init__()
        self.device = device
        self.n_channels = n_channels

        # self.fc0 = nn.Linear(n_channels * 4, hidden_size)
        self.conv3d = nn.Conv3d(n_channels * 4, hidden_size, kernel_size=(3, 3, 3), padding=1)
        self.fc1 = nn.Linear(hidden_size, n_channels, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.fire_rate = fire_rate
        self.to(self.device)
        self.dx, self.dy, self.dz = [ item.to(self.device) for item in NCAModel3D.sobel_filter_3d() ]

    def load_pretrained(self, path):
        self.load_state_dict(torch.load(path))

    @staticmethod
    def sobel_filter_3d():
        dx = torch.outer(torch.tensor([1, 2, 1], dtype=torch.float32),
                         torch.tensor([-1, 0, 1], dtype=torch.float32))
        dx = dx.repeat([3, 1, 1]) * torch.tensor([1, 2, 1]).reshape(3, 1, 1)
        dx = dx / dx.abs().sum()
        dy = dx.transpose(1, 2)
        dz = dx.transpose(0, 2)
        return dx.permute(1, 2, 0), dy.permute(1, 2, 0), dz.permute(1, 2, 0)

    @staticmethod
    def alive(x):
        return F.max_pool3d(x[:, 3:4, :, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x):

        def _perceive_with(x_, conv_weights):
            conv_weights = conv_weights.view(1, 1, 3, 3, 3).repeat(self.n_channels, 1, 1, 1, 1)
            return F.conv3d(x_, conv_weights, padding=1, groups=self.n_channels)

        y1 = _perceive_with(x, self.dx)
        y2 = _perceive_with(x, self.dy)
        y3 = _perceive_with(x, self.dz)
        y = torch.cat((x, y1, y2, y3), dim=1)
        return y

    def update(self, x, fire_rate):

        x = x.transpose(1, 4)

        pre_life_mask = self.alive(x)

        dx = self.perceive(x)

        dx = self.conv3d(dx)
        dx = dx.transpose(1, 4)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        fire_rate = self.fire_rate if fire_rate is None else fire_rate

        stochastic = torch.rand([dx.size(0),
                                 dx.size(1),
                                 dx.size(2),
                                 dx.size(3), 1], dtype=torch.float32) > fire_rate

        stochastic = stochastic.to(self.device)
        dx = dx * stochastic

        x = x + dx.transpose(1, 4)

        post_life_mask = self.alive(x)

        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask

        return x.transpose(1, 4)

    def forward(self, x, steps=1, fire_rate=None):
        for step in range(steps):
            x = self.update(x, fire_rate)
        return x