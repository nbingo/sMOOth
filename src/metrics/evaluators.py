from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm


class ClassificationAcc(DatasetEvaluator):
    def __init__(self):
        super().__init__()
        self.corr = 0
        self.total = 0

    def reset(self):
        self.corr = self.total = 0

    def process(self, inputs, outputs):
        labels = inputs['label'].to(dtype=int, device='cpu')
        self.corr += (outputs.argmax(dim=1).cpu() == labels.cpu()).sum().item()
        self.total += len(labels)

    def evaluate(self):
        all_corr_total = comm.all_gather([self.corr, self.total])
        corr = sum(x[0] for x in all_corr_total)
        total = sum(x[1] for x in all_corr_total)
        return {"accuracy": corr / total}
