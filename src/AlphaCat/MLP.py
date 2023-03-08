import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class DeepQMLP(nn.Module):

    device = torch.device("cuda")

    def __init__(self, size: int, lr: float=0.9):
        super().__init__()
        self.size = size
        self.input = nn.Linear(size ** 2 * 2, size ** 2 * 2)
        self.hide1 = nn.Linear(size ** 2 * 2, size ** 2)
        self.hide2 = nn.Linear(size ** 2, size ** 2)
        self.output = nn.Linear(size ** 2, size ** 2)

    def forward(self, x: np.array) -> torch.tensor:
        # print("i_o", x.shape)
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        else:
            x = x.clone().detach()
        # print("i_n", x.shape)
        x = F.relu(self.input(x))
        x = F.relu(self.hide1(x))
        x = F.relu(self.hide2(x))
        return self.output(x)

    def save(self, fname='Model.pth'):
        folder = os.path.dirname(os.path.abspath(__file__))
        torch.save(self.state_dict(), folder+"\\"+fname)
