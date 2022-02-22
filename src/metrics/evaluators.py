from __future__ import annotations

from detectron2.evaluation import DatasetEvaluator, DatasetEvaluators
from detectron2.utils import comm
from detectron2.config import LazyConfig, instantiate

import torch

from .losses import _binary_equalized_odds_viol, _compute_binary_equalized_odds_counters


class MultiObjectiveEvaluator(DatasetEvaluator):
    def __init__(self, dataset_evaluators: DatasetEvaluators):
        super().__init__()

        self.evaluators = instantiate(dataset_evaluators) \
            if isinstance(dataset_evaluators, LazyConfig) else dataset_evaluators
        self.num_evaluators = len(self.evaluators._evaluators)

    def reset(self):
        self.evaluators.reset()

    def process(self, inputs, outputs):
        self.evaluators.process(inputs, outputs)

    def evaluate(self):
        # Choose a central preference vector
        # TODO: a central preference vector of course isn't always the best choice, so going to need to do hypervolume
        #  or something like that sometime soon. Really honestly this kind of cumulative evaluator doesn't work in this
        #  case and has to be hypervolume
        preference_ray = torch.sqrt(torch.ones(self.num_evaluators) / self.num_evaluators)
        results = self.evaluators.evaluate()
        # Make into torch tensor
        results = torch.Tensor(list(results.values()))
        result = torch.matmul(results, preference_ray)
        return result


class ClassificationAcc(DatasetEvaluator):
    def __init__(self):
        super().__init__()
        self.corr = 0
        self.total = 0

    def reset(self):
        self.corr = self.total = 0

    def process(self, inputs: dict[str, torch.Tensor], outputs: torch.Tensor):
        labels = inputs['label'].to(dtype=int, device='cpu')
        self.corr += (outputs.argmax(dim=1).cpu() == labels.cpu()).sum().item()
        self.total += len(labels)

    def evaluate(self):
        all_corr_total = comm.all_gather([self.corr, self.total])
        corr = sum(x[0] for x in all_corr_total)
        total = sum(x[1] for x in all_corr_total)
        return {"accuracy": corr / total}


class BinaryEqualizedOddsViolation(DatasetEvaluator):
    def __init__(self):
        super().__init__()
        # Create a counter to compute probability of positive classification across groups and true label
        # Format is [pred, label, group], so self.cong_prob_counters[0,1] = number of examples seen so far with true
        # label 0 belonging to group 1 and predicted as label 1 (we are only keeping track of positive predictions here
        # since it's binary classification with binary groups, but will generalize later)
        self.cond_prob_counters = torch.zeros((2, 2, 2), dtype=torch.long)

    def reset(self):
        self.cond_prob_counters = torch.zeros((2, 2, 2), dtype=torch.long)

    def process(self, inputs: dict[str, torch.Tensor], outputs: torch.Tensor):
        _compute_binary_equalized_odds_counters(inputs, outputs, self.cond_prob_counters)

    def evaluate(self):
        return {'Equalized odds violation': _binary_equalized_odds_viol(self.cond_prob_counters)}
