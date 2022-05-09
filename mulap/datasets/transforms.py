import random
from torch import nn


class RandomCrop(nn.Module):
    def __init__(self, crop_size):
        super().__init__()
        self.crop_size = crop_size

    def forward(self, x):
        random_int_start = random.randint(0, x.size(1) - self.crop_size)
        x = x[:, random_int_start:random_int_start+self.crop_size]
        return x
