# adapted from https://github.com/BigRedT/deep_income/blob/master/model.py

import torch.nn.functional as F
import torch.nn as nn
from models.base import BaseModel


class IncomeClassifierConstants():
    def __init__(self):
        self.in_dim = 105
        self.hidden_dim = 105
        self.num_hidden_blocks = 2
        self.drop_prob = 0.2
        self.out_dim = 2


class IncomeClassifier(BaseModel):
    def __init__(self, in_dim, hidden_dim, num_hidden_blocks, drop_prob, out_dim, loss_fn, device='cuda'):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_blocks = num_hidden_blocks
        self.drop_prob = drop_prob
        self.out_dim = out_dim
        self.loss_fn = loss_fn
        self.device = device

        if self.num_hidden_blocks == 0:
            self.layers = nn.Linear(self.in_dim, self.out_dim)

        else:
            # Add input layers
            layers = [self.input_block()]

            # Add hidden layers
            for i in range(self.num_hidden_blocks):
                layers.append(self.hidden_block())

            # Add output layers
            layers.append(self.output_block())

            self.layers = nn.Sequential(*layers)

        self.softmax_layer = nn.Softmax(1)

    def input_block(self):
        return nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(self.drop_prob),
            nn.Sigmoid())

    def hidden_block(self):
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(self.drop_prob),
            nn.Sigmoid())

    def output_block(self):
        return nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, data):
        features = data['feat'].to(self.device)
        labels = data['label'].to(dtype=int, device=self.device)
        logits = self.layers(features)
        # probs = self.softmax_layer(logits)
        if self.training:
            return self.loss_fn(logits, labels)
        else:
            return logits
