# adapted from https://github.com/BigRedT/deep_income/blob/master/dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from data.Adult.income_attributes import ATTRS


class FeatDataset(Dataset):
    def __init__(self, subset: str, income_const: dict):
        self.subset = subset
        self.feats, self.labels, self.sample_ids = self.load_data(income_const)
        # Get group membership by converting the female label into boolean values
        self.groups = (np.sign(self.feats[:, ATTRS.index('Female')]) - 1).astype(bool)

    def load_data(self, income_const: dict):
        if self.subset == 'test':
            subset_npy = 'test_npy'
        else:
            subset_npy = 'train_val_npy'

        feats = np.load(os.path.join(
            income_const['proc_dir'],
            income_const[subset_npy]['feat']))
        labels = np.load(os.path.join(
            income_const['proc_dir'],
            income_const[subset_npy]['label']))
        sample_ids = np.load(os.path.join(
            income_const['proc_dir'],
            income_const['sample_ids_npy'][self.subset]))

        return feats, labels, sample_ids

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, i):
        idx = self.sample_ids[i]

        to_return = {
            'feat': self.feats[idx],
            'label': self.labels[idx],
            'group': self.groups[idx],
        }

        return to_return
