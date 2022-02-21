from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm
import torch


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
        labels = inputs['label'].to(dtype=int, device='cpu')
        groups = inputs['group'].to(dtype=int, device='cpu')
        _, pred = outputs.max(dim=1)
        pred = pred.to(device='cpu')
        for label in [0, 1]:     # loop over true labels
            for group in [0, 1]:     # loop over groups
                for pred_label in [0, 1]:       # loop over predicted labels
                    # Sum predictions that match the true label and group, and then only those with the desired
                    # predicted label
                    self.cond_prob_counters[pred_label, label, group] += \
                        (pred[(labels == label) & (groups == group)] == pred_label).sum()

    def evaluate(self):
        # Compute prob of positive result for each possibility of group and true label
        probs = self.cond_prob_counters[1, :, :] - self.cond_prob_counters.sum(dim=0)
        # Compute equalized odds violation by first taking differences across group and then summing across true label
        eq_odds_viol = (probs[:, 0] - probs[:, 1]).abs().sum()

        return {'Equalized odds violation': eq_odds_viol}
