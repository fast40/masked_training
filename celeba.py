import pathlib

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=Compose([CenterCrop((128, 128)), ToTensor()])):
        self.image_paths = list(pathlib.Path(root_dir).glob('*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


dataset = ImageDataset('celeba')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
