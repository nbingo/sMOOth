import time
import torch
from detectron2.engine.train_loop import SimpleTrainer


class LinearScalarizationTrainer(SimpleTrainer):
    def __init__(self, model, data_loader, optimizer, preference_vector: torch.Tensor = torch.ones(2) / 2):
        super().__init__(model, data_loader, optimizer)

        self.preference_vector = preference_vector
        print(f'Running linear scalarization method with preference vector: {self.preference_vector}')

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
        losses = torch.matmul(torch.stack(list(loss_dict.values())), self.preference_vector)
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

class CalibratedLinearScalarizationTrainer(SimpleTrainer):
    def __init__(self, model, data_loader, optimizer, preference_vector: torch.Tensor = torch.ones(2) / 2):
        super().__init__(model, data_loader, optimizer)

        self.preference_vector = preference_vector
        print(f'Running calibrated linear scalarization method with preference vector: {self.preference_vector}')

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
        task_losses = torch.stack(list(loss_dict.values()))
        losses = torch.matmul(task_losses, self.preference_vector)
        cossim = torch.nn.functional.cosine_similarity(task_losses, self.preference_vector, dim=0)
        # Essentially assuming labmda (i.e., weight facator) = 1
        losses -= cossim
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

