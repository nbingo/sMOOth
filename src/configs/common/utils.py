import torch
from detectron2.data.samplers import TrainingSampler, InferenceSampler


def build_data_loader(dataset, batch_size, num_workers, training=True):
    return torch.utils.data.DataLoader(
        dataset,
        sampler=(TrainingSampler if training else InferenceSampler)(len(dataset)),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
