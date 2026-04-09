
import torch.nn as nn


class Logistic(nn.Module):
    def __init__(self, in_dim=784, out_dim=10):
        # for mnist
        super(Logistic, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = x.reshape(-1, 784)
        logit = self.layer(x)
        return logit