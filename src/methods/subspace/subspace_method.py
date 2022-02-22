import time
import torch
from torch.distributions.dirichlet import Dirichlet

from detectron2.engine.train_loop import SimpleTrainer


class SubspaceTrainer(SimpleTrainer):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, model, data_loader, optimizer, alpha: float):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
            alpha: Parameter for Dirichlet distribution
        """
        super().__init__(model, data_loader, optimizer)

        self.dirichlet_dist = Dirichlet(concentration=alpha)

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        # Sample from Dirichlet
        preference_vector = self.dirichlet_dist.sample()
        losses = torch.matmul(torch.Tensor(loss_dict.values()), preference_vector)
        # TODO: Create subspace model class to make this standard or general MOO method class that requires preference
        #  vector during inference. In general shouldn't be needing to import or reference specifics
        #  from out of library code
        # TODO: Will also need to change specific dataset evaluators that use MOO methods that require preference
        #  vectors for inference in a similar way
        loss_dict = self.model(data, preference_vector)
        loss_dict['total_loss'] = losses

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

