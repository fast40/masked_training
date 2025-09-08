import pathlib

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = list(pathlib.Path(root_dir).glob('*.jpg'))
        self.transform = transform if transform is not None else Compose([CenterCrop((128, 128)), ToTensor()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx) -> torch.Tensor:
        image = Image.open(self.image_paths[idx]).convert('RGB')

        return self.transform(image)
    


dataset = ImageDataset('celeba')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
