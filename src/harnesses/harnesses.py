# Adapted from Detectron2: https://github.com/facebookresearch/detectron2/blob/main/tools/lazyconfig_train_net.py
import logging

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    default_setup,
    default_writers,
    hooks,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm



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

# Base class for trainer harnesses


class BaseHarness():
    def __init__(self, args):
        self.args = args
        self.cfg = LazyConfig.load(self.args.config_file)
        self.cfg = LazyConfig.apply_overrides(self.cfg, self.args.opts)
        default_setup(self.cfg, self.args)

    def main(self):
        if self.args.eval_only:
            self.do_test()
        else:
            self.do_train()

    def do_test(self):
        model = instantiate(self.cfg.model)
        model.to(self.cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(self.cfg.train.init_checkpoint)
        print(self.do_test())
        if "evaluator" in self.cfg.dataloader:
            ret = inference_on_dataset(
                model, instantiate(self.cfg.dataloader.test), instantiate(self.cfg.dataloader.evaluator)
            )
            print_csv_format(ret)
            print(ret)
            return ret

    def do_train(self):
        """
        Args:
            cfg: an object with the following attributes:
                model: instantiate to a module
                dataloader.{train,test}: instantiate to dataloaders
                dataloader.evaluator: instantiate to evaluator for test set
                optimizer: instantaite to an optimizer
                lr_multiplier: instantiate to a fvcore scheduler
                train: other misc config defined in `configs/common/train.py`, including:
                    output_dir (str)
                    init_checkpoint (str)
                    amp.enabled (bool)
                    max_iter (int)
                    eval_period, log_period (int)
                    device (str)
                    checkpointer (dict)
                    ddp (dict)
        """
        model = instantiate(self.cfg.model)
        logger = logging.getLogger("detectron2")
        logger.info("Model:\n{}".format(model))
        model.to(self.cfg.train.device)

        self.cfg.optimizer.params.model = model
        optim = instantiate(self.cfg.optimizer)

        train_loader = instantiate(self.cfg.dataloader.train)

        model = create_ddp_model(model, **self.cfg.train.ddp)
        trainer = self.cfg.train.trainer(model, train_loader, optim)
        checkpointer = DetectionCheckpointer(
            model,
            self.cfg.train.output_dir,
            trainer=trainer,
        )
        trainer.register_hooks(
            [
                hooks.IterationTimer(),
                hooks.LRScheduler(scheduler=instantiate(self.cfg.lr_multiplier)),
                hooks.PeriodicCheckpointer(checkpointer, **self.cfg.train.checkpointer)
                if comm.is_main_process()
                else None,
                hooks.EvalHook(self.cfg.train.eval_period, lambda: self.do_test()),
                hooks.PeriodicWriter(
                    default_writers(self.cfg.train.output_dir, self.cfg.train.max_iter),
                    period=self.cfg.train.log_period,
                )
                if comm.is_main_process()
                else None,
            ]
        )

        checkpointer.resume_or_load(self.cfg.train.init_checkpoint, resume=self.args.resume)
        if self.args.resume and checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            start_iter = trainer.iter + 1
        else:
            start_iter = 0
        trainer.train(start_iter, self.cfg.train.max_iter)




