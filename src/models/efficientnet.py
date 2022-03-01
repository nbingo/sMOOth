import torchvision.models as models
from src.models.base import BaseModel

class EfficientNetB4(BaseModel):
    def __init__(self, loss_fn, pretrained: bool = False):
        super().__init__()
        self.model = models.efficientnet_b4(pretrained=pretrained)
        self.loss_fn = loss_fn

    def forward(self, data):
        feats = data['feat']

        logits = self.model(feats)
        if self.training:
            return self.loss_fn(data, logits)
        return logits
