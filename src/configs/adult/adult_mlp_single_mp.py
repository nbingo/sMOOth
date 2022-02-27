# won't be editing all of these, but we need them to be imported so that they can be loaded in the config
from .adult_mlp_standard import (
    dataloader,
    train,
    model,
    optimizer,
    lr_multiplier
)

from src.harnesses.harnesses import MultiProcessHarness

train.gpus = [0]
train.process_over_vals = [1e-3]
train.process_over_key = 'optimizer.lr'
train.harness = MultiProcessHarness
