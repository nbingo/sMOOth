# Adapted from Detectron2: https://github.com/facebookresearch/detectron2/blob/main/tools/lazyconfig_train_net.py

"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import torch.autograd
from detectron2.engine import (
    default_argument_parser,
    launch,
)
from detectron2.config import LazyConfig
from src.harnesses.harnesses import SimpleHarness


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    args = default_argument_parser().parse_args()
    cfg = LazyConfig.load(args.config_file)
    train_harness = cfg.train.harness(args)
    launch(
        train_harness.main(),
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )