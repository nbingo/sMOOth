from __future__ import annotations

import torch
from typing import Callable, Iterable
from detectron2.config import LazyConfig, instantiate
from collections import OrderedDict


class MultiObjectiveLoss:
    def __init__(self, losses: Iterable[Callable | LazyConfig]):
        # Instantiate the loss function if needed
        self.losses = [instantiate(loss) if isinstance(loss, LazyConfig) else loss for loss in losses]

    def __call__(self, inputs: dict[str, torch.Tensor], outputs):
        loss_dict = OrderedDict()
        # Simply compute all the losses and store them
        for loss in self.losses:
            loss_dict[loss.__name__] = loss(inputs, outputs)
        return loss_dict


def _binary_equalized_odds_viol(cond_prob_counters: torch.Tensor):
    # Compute prob of positive result for each possibility of group and true label
    probs = cond_prob_counters[1, :, :] - cond_prob_counters.sum(dim=0)
    # Compute equalized odds violation by first taking differences across group and then summing across true label
    # Then normalize by total number of examples
    eq_odds_viol = (probs[:, 0] - probs[:, 1]).abs().sum() / cond_prob_counters.sum()

    return eq_odds_viol


def _compute_binary_equalized_odds_counters(inputs: dict[str, torch.Tensor], outputs,
                                            cond_prob_counters: torch.Tensor = None):
    """
    Computers conditional probability counters for use in equalized odds violation loss
    :param inputs: The same inputs as given to the model in dictionary format
    :param outputs: The outputs of the model
    :param cond_prob_counters: Tensor of shape (2, 2, 2) to keep track of conditional counters
    :return: cond_prob_counters filled in with the correct counts
    """
    if cond_prob_counters is None:
        cond_prob_counters = torch.zeros(2, 2, 2)
    labels = inputs['label'].to(dtype=int, device='cpu')
    groups = inputs['group'].to(dtype=int, device='cpu')
    _, pred = outputs.max(dim=1)
    pred = pred.to(device='cpu')
    # Create a counter to compute probability of positive classification across groups and true label
    # Format is [pred, label, group], so self.cong_prob_counters[0,1] = number of examples seen so far with true
    # label 0 belonging to group 1 and predicted as label 1 (we are only keeping track of positive predictions here
    # since it's binary classification with binary groups, but will generalize later)
    cond_prob_counters = torch.zeros((2, 2, 2), dtype=torch.long)
    for label in [0, 1]:  # loop over true labels
        for group in [0, 1]:  # loop over groups
            for pred_label in [0, 1]:  # loop over predicted labels
                # Sum predictions that match the true label and group, and then only those with the desired
                # predicted label
                cond_prob_counters[pred_label, label, group] += \
                    (pred[(labels == label) & (groups == group)] == pred_label).sum()
    return cond_prob_counters


def equalized_odds_violation(inputs: dict, outputs):
    cond_prob_counters = _compute_binary_equalized_odds_counters(inputs, outputs)
    return _binary_equalized_odds_viol(cond_prob_counters)
