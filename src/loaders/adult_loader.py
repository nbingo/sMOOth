import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class FeatDataset(Dataset):
    def __init__(self, subset: str, income_const: dict):
        self.subset = subset
        self.feats, self.labels, self.sample_ids = self.load_data()
        
        if self.subset=='test':
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

    def __getitem__(self,i):
        idx = self.sample_ids[i]
        
        to_return = {
            'feat': self.feats[idx],
            'label': self.labels[idx],
        }

        return to_return