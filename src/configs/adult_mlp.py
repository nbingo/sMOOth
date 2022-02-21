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
from detectron2.config import LazyCall as L
from detectron2.model_zoo import get_config
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm

from configs.common.utils import build_data_loader
from src.models.adult_mlp import IncomeClassifier

from data.Adult.dataset import FeatDataset


"""
Note: Here we put reusable code (models, evaluation, data) together with configs just as a
proof-of-concept, to easily demonstrate what's needed to train a ImageNet classifier in detectron2.
Writing code in configs offers extreme flexibility but is often not a good engineering practice.
In practice, you might want to put code in your project and import them instead.
"""


class ClassificationAcc(DatasetEvaluator):
    def __init__(self):
        super().__init__()
        self.corr = 0
        self.total = 0

    def reset(self):
        self.corr = self.total = 0

    def process(self, inputs, outputs):
        _, label = inputs
        self.corr += (outputs.argmax(dim=1).cpu() == label.cpu()).sum().item()
        self.total += len(label)

    def evaluate(self):
        all_corr_total = comm.all_gather([self.corr, self.total])
        corr = sum(x[0] for x in all_corr_total)
        total = sum(x[1] for x in all_corr_total)
        return {"accuracy": corr / total}


# --- End of code that could be in a project and be imported


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

dataloader.evaluator = L(ClassificationAcc)()

model = L(IncomeClassifier)(
    in_dim=105,
    hidden_dim=105,
    num_hidden_blocks=2,
    drop_prob=0.2,
    out_dim=2,
    loss_fn=F.cross_entropy
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


train = get_config("common/train.py").train
train.init_checkpoint = None
# max_iter = number epochs * (train dataset size / batch size)
train.max_iter = 50 * 30162 // 256
train.eval_period = 30162 // 256

