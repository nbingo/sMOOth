# won't be editing all of these, but we need them to be imported so that they can be loaded in the config
from .adult_mlp_standard import (
    dataloader,
    train,
    model,
    optimizer,
    lr_multiplier
)

from src.metrics.losses import MultiObjectiveLoss, cross_entropy_loss, equalized_odds_violation
from src.methods import LinearScalarizationTrainer
from src.harnesses import LinearScalarizationHarness

from detectron2.config import LazyCall as L

model.loss_fn = L(MultiObjectiveLoss)(losses=[cross_entropy_loss, equalized_odds_violation])

# Options for linear scalarization. Mainly for get_reference_directions from PYMOO
train.num_preference_vector_partitions = 4
train.preference_ray_idx = 2     # Which indexed preference vector to use
train.gpus = [0]
train.trainer = LinearScalarizationTrainer
train.harness = LinearScalarizationHarness
