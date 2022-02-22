"""
An example config file to train a ImageNet classifier with detectron2.
Model and dataloader both come from torchvision.
This shows how to use detectron2 as a general engine for any new models and tasks.

To run, use the following command:

python tools/lazyconfig_train_net.py --config-file configs/Misc/torchvision_imagenet_R_50.py \
    --num-gpus 8 dataloader.train.dataset.root=/path/to/imagenet/

"""

import yaml
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from fvcore.common.param_scheduler import CosineParamScheduler

from detectron2.solver import WarmupParamScheduler
from detectron2.solver.build import get_default_optimizer_params
from detectron2.config import LazyCall as L, LazyConfig
from detectron2.evaluation import DatasetEvaluators

from src.configs.common.utils import build_data_loader
from src.models.adult_mlp import IncomeClassifier
from src.metrics.evaluators import ClassificationAcc, BinaryEqualizedOddsViolation, MultiObjectiveEvaluator
from src.metrics.losses import MultiObjectiveLoss, equalized_odds_violation
from src.loaders.adult_loader import FeatDataset
from src.methods.subspace.subspace_wrapper import to_subspace_class
from src.methods.subspace.subspace_method import SubspaceTrainer


dataloader = OmegaConf.create()
dataloader.train = L(build_data_loader)(
    dataset=L(FeatDataset)(
        subset='train',
        income_const=yaml.load(open('/lfs/local/0/nomir/sMOOth/data/Adult/income.yml'), Loader=yaml.FullLoader)
    ),
    batch_size=256,
    num_workers=4,
    training=True,
)

dataloader.test = L(build_data_loader)(
    dataset=L(FeatDataset)(
        subset='val',
        income_const=yaml.load(open('/lfs/local/0/nomir/sMOOth/data/Adult/income.yml'), Loader=yaml.FullLoader)
    ),
    batch_size=256,
    num_workers=4,
    training=False,
)

dataloader.evaluator = L(MultiObjectiveEvaluator)(
    dataset_evaluators=L(DatasetEvaluators)(
        evaluators=(ClassificationAcc(), BinaryEqualizedOddsViolation())
    )
)

train = LazyConfig.load("/lfs/local/0/nomir/sMOOth/src/configs/common/train.py").train
train.init_checkpoint = None
# max_iter = number epochs * (train dataset size / batch size)
train.max_iter = 50 * 30162 // 256
train.eval_period = 30162 // 256
train.trainer = SubspaceTrainer

model = L(to_subspace_class(model_class=IncomeClassifier, num_vertices=2))(
    in_dim=105,
    hidden_dim=105,
    num_hidden_blocks=2,
    drop_prob=0.2,
    out_dim=2,
    loss_fn=MultiObjectiveLoss([F.cross_entropy, equalized_odds_violation]),
    device=train.device,
)


optimizer = L(torch.optim.Adam)(
    params=L(get_default_optimizer_params)(),
    lr=1e-3,
    weight_decay=1e-4,
)

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(CosineParamScheduler)(
        start_value=0.1,
        end_value=1e-4,
    ),
    warmup_length=1 / 100,
    warmup_factor=0.1,
)




