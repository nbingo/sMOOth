"""
An example config file to train a ImageNet classifier with detectron2.
Model and dataloader both come from torchvision.
This shows how to use detectron2 as a general engine for any new models and tasks.

To run, use the following command:

python tools/lazyconfig_train_net.py --config-file configs/Misc/torchvision_imagenet_R_50.py \
    --num-gpus 8 dataloader.train.dataset.root=/path/to/imagenet/

"""

import torch
from omegaconf import OmegaConf
from fvcore.common.param_scheduler import CosineParamScheduler

from detectron2.solver import WarmupParamScheduler
from detectron2.solver.build import get_default_optimizer_params
from detectron2.config import LazyConfig, LazyCall as L
from detectron2.evaluation import DatasetEvaluators

from src.configs.common.utils import build_data_loader
from src.loaders import WildsFMoWDataset
from src.metrics.evaluators import ClassificationAcc, BinaryEqualizedOddsViolation
from src.models import EfficientNetB4
from src.metrics.losses import cross_entropy_loss

dataloader = OmegaConf.create()
dataloader.train = L(build_data_loader)(
    dataset=L(WildsFMoWDataset)(
        subset='train',
    ),
    batch_size=128,
    num_workers=4,
    training=True,
)

dataloader.test = L(build_data_loader)(
    dataset=L(WildsFMoWDataset)(
        subset='val',
    ),
    batch_size=128,
    num_workers=4,
    training=False,
)

# Can also be list of DatasetEvaluators
dataloader.evaluator = L(DatasetEvaluators)(evaluators=(ClassificationAcc()))

train = LazyConfig.load("/lfs/local/0/nomir/sMOOth/src/configs/common/train.py").train
train.init_checkpoint = None
# max_iter = number epochs * (train dataset size / batch size)
train.max_iter = 10 * 30162 // 256
train.eval_period = 30162 // 256
train.output_dir = './output/fmow/standard'

model = L(EfficientNetB4)(
    pretrained=True,
    loss_fn=cross_entropy_loss,
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
