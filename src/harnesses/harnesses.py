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

from multiprocessing import Pool
from itertools import cycle
from copy import deepcopy

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


class SimpleHarness:
    """
    Simple training harness for single process and model training
    """

    def __init__(self, args, cfg=None):
        self.args = args
        if cfg is None:
            self.cfg = LazyConfig.load(self.args.config_file)
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
            self.cfg = LazyConfig.apply_overrides(self.cfg, self.args.opts)
            default_setup(self.cfg, self.args)
        else:       # We assume cfg is properly made
            self.cfg = cfg

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
        if "evaluator" in self.cfg.dataloader:
            ret = inference_on_dataset(
                model, instantiate(self.cfg.dataloader.test), instantiate(self.cfg.dataloader.evaluator)
            )
            print_csv_format(ret)
            print(ret)
            return ret

    def do_train(self):

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


class MultiProcessHarness(SimpleHarness):
    """
    its config file must have the extra parameter train.process_over_key key with value indicating the key in the config
    file that will be changed in each model that is spawned. In addition, it must have train.process_over_vals
    which is a list of values that will be used to replace the key indicated in train.process_over_key in each of the
    individually spawned models.
    It also must have train.num_workers to inidcate the maximum number of workers allowed to be used (CPU workers.
    Lastly, it must have train.gpus and train.num_models_per_gpu, a list of GPUs to use that will be iterated over to
    train the models with at most train.num_models_per_gpu models on each gpu
    """

    def __init__(self, args):
        super().__init__(args)
        # Create the new configs that will be used for the various spawned proceses
        self.modified_cfgs = []
        cycle_gpus = cycle(self.cfg.train.gpus)
        for val in self.cfg.train.process_over_vals:
            new_cfg = deepcopy(self.cfg)
            new_cfg[self.cfg.train.process_over_key] = val        # TODO: This probably isn't the right syntax here...
            new_cfg.train.device = f'cuda:{next(cycle_gpus)}'
            self.modified_cfgs.append(new_cfg)

    # TODO: Create the do_train and do_test methods that actually end up spawning the subprocesses.
    #  In this probably have to make sure they all have separate loggers somehow...
    #  And also probably need ot make a shared queue for using the correct gpu.

    def _init_harness_do_test(self, cfg):
        harness = SimpleHarness(self.args, cfg)
        harness.do_test()

    def _init_harness_do_train(self, cfg):
        harness = SimpleHarness(self.args, cfg)
        harness.do_train()

    def do_test(self):
        with Pool(processes=len(self.cfg.train.gpus)) as pool:
            pool.imap(self._init_harness_do_test, self.modified_cfgs)

    def do_train(self):
        with Pool(processes=len(self.cfg.train.gpus)) as pool:
            pool.imap(self._init_harness_do_train, self.modified_cfgs)