from infra import *

import pathlib

import torch
from torchvision.transforms import Compose, CenterCrop, ToTensor
from PIL import Image


class ImageDataset:
    def __init__(self, root_dir='celeba', batch_size=10):
        self.batch_size = batch_size
        logger.info(f'Creating ImageDataset with batch_size={self.batch_size}')

        logger.info('Loading all filenames into dataset object...')
        self.image_paths = list(pathlib.Path(root_dir).glob('*.jpg'))
        logger.info('Done loading all filenames into dataset object.')
        self.transform = Compose([CenterCrop((128, 128)), ToTensor()])
        self.len = len(self.image_paths)
        self.index = 0

    def __getitem__(self, idx) -> torch.Tensor:
        image = Image.open(self.image_paths[idx]).convert('RGB')

        return self.transform(image)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index + self.batch_size > self.len:
            logger.info(f'ImageDataset reached index {self.index} (last index is {self.len - 1}). Could not create a batch of size {self.batch_size} at this location, so wrapping self.index back to 0')
            self.index = 0

        batch = torch.stack([self[self.index + i] for i in range(self.batch_size)])

        self.index += self.batch_size

        return batch
