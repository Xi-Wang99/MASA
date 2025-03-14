import torch.nn as nn
import torch

class net(nn.Module):
    def __init__(self, in_features, features):
        super().__init__()
        n = len(features)
        layer1 = nn.Linear(in_features, features[0])
        list = [nn.Linear(features[i], features[i+1]) for i in range(n-1)]
        list.insert(0, layer1)
        self.model = nn.Sequential(*list)
    def forward(self, x):
        y = self.model(x)
        # print(y.shape)
        return y


class mlp(nn.Module):
    def __init__(self, in_features, features):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features, features[0]))
        layers.append(nn.ReLU())
        for i in range(len(features) - 1):
            layers.append(nn.Linear(features[i], features[i+1]))
            layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        y = self.model(x)
        return y


class attention(nn.Module):
    def __init__(self, in_features, features):
        super().__init__()
        n = len(features)
        layers = [nn.Flatten(), nn.Linear(in_features, features[0]), nn.ReLU()]
        layers += [item for sublist in [[nn.Linear(features[i], features[i+1]), nn.ReLU()] for i in range(n-1)] for item in sublist]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        y = self.model(x)
        return y



