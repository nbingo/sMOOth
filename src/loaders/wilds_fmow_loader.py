from wilds import get_dataset
from wilds.common.grouper import CombinatorialGrouper
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class WildsFMoWDataset(Dataset):
    def __init__(self, subset: str):
        self.full_dataset = get_dataset(dataset="fmow", download=True)
        # Need to normalize like this for pre-trained efficient net from pytorch
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.dataset = self.full_dataset.get_subset(
            subset,
            transform=transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor(), self.normalize]),
        )
        print(f'Using WILDS FMoW Dataset on {subset} split with {len(self.dataset)} examples')
        self.grouper = CombinatorialGrouper(self.full_dataset, ['region'])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        feat, label, metadata = self.dataset[item]
        # Need to pad dimension of metadata for grouper if only one item was requested
        if len(metadata.size()) == 1:
            metadata = metadata.unsqueeze(0)
        group = self.grouper.metadata_to_group(metadata).squeeze()
        return {
            'label': label,
            'feat': feat,
            'group': group
        }

