import torch

import numpy as np
import torchvision.transforms.functional as F

from torchvision import transforms, datasets
from torch.utils.data import Dataset

from codefiles.datasets.utils import create_missing_data_masks, apply_missing_mask

class HalveData(object):

    def __init__(
            self, 
            split: str ="horizontal",
            dataset: str ="fmnist",
            subset: bool = False,
            unpaired: bool = False,
            missing: bool = False
    ) -> None: 
        self.split = split
        self.unpaired = unpaired 
        self.missing = missing

        if dataset == "mnist":
            self.transform_top = transforms.Compose([
                transforms.Normalize((0.0503,), (0.2009,))
            ])
            self.transform_bot = transforms.Compose([
                transforms.Normalize((0.0217,), (0.1347,))
            ])
            if subset:
                self.transform_top = transforms.Compose([
                    transforms.Normalize((0.1230,), (0.2999,))
                ])
                self.transform_bot = transforms.Compose([
                    transforms.Normalize((0.1375,), (0.3150,))
                ])
        elif dataset == "fashion_mnist":
            self.transform_top = transforms.Compose([
                transforms.Normalize((0.2581,), (0.3452,))
            ])
            self.transform_bot = transforms.Compose([
                transforms.Normalize((0.3140,), (0.3586,))
            ])

    def __call__(self, 
                 sample
    ): 
        if self.split == "horizontal":
            data_top = sample[:, :sample.shape[1]//2, :]
            data_bot = sample[:, sample.shape[1]//2:, :]
            
            data_top = self.transform_top(data_top)
            data_bot = self.transform_bot(data_bot)

            if self.missing: 
                data_bot = torch.zeros_like(data_bot)
            
            elif self.unpaired:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
                self.unpaired_ds = datasets.FashionMNIST(root = 'data', train = True, transform = transform, download = True)
                randint = np.random.randint(0, len(self.unpaired_ds))
                data_bot = self.unpaired_ds[0][randint][sample.shape[1]//2:, :]

        if self.split == "vertical":
            data_top = sample[:, :, :sample.shape[2]//2]
            data_bot = sample[:, :, sample.shape[2]//2:]

            data_top = self.transform_top(data_top)
            data_bot = self.transform_bot(data_bot)

        return data_top, data_bot

class Halved_Fashion_or_Vanilla_MNIST(Dataset):
    def __init__(
            self, 
            root: str = "",  
            zero_fill_rates: list = [0.0, 0.0],
            train: bool = True, 
            num_modalities: int = 2,
            download: bool = False, 
            variant: str = "unimodal_1",
            transform = None,
            dataset: str = "fmnist",
            seed: int = 42
    ) -> None:
        self.num_modalities = num_modalities
        self.variant = variant
        
        # Initialize datasets
        if dataset == "mnist":
            self.dataset = datasets.MNIST(root=root, train=train, transform=transform, download=download)
        if dataset == "fmnist":
            self.dataset = datasets.FashionMNIST(root=root, train=train, transform=transform, download=download)

        # normalization
        self.transform_top = transforms.Compose([
            transforms.Normalize((0.2581,), (0.3452,))
        ])  # no missing top 
        self.transform_bot = transforms.Compose([
            transforms.Normalize((0.3139,), (0.3585,))
        ])  # no missing bot

        # missing data 
        self.zero_fill_masks, self.missing_stats = create_missing_data_masks(
            total_samples=len(self.dataset),
            num_modalities=self.num_modalities,
            missing_rates=zero_fill_rates,
            random_seed=seed
        )

    def get_normalization_params(self):
        top_sum = torch.zeros(1)
        top_sum_sq = torch.zeros(1)
        bot_sum = torch.zeros(1)
        bot_sum_sq = torch.zeros(1)
        top_count = 0
        bot_count = 0

        for idx, (img, _) in enumerate(self.dataset):
            if not isinstance(img, torch.Tensor):
                img = transforms.ToTensor()(img)

            data_top = img[:, :img.shape[1]//2, :]
            data_bot = img[:, img.shape[1]//2:, :]

            if idx not in self.zero_fill_indices[0]:
                top_sum += data_top.mean()
                top_sum_sq += (data_top ** 2).mean()
                top_count += 1

            if idx not in self.zero_fill_indices[1]:
                bot_sum += data_bot.mean()
                bot_sum_sq += (data_bot ** 2).mean()
                bot_count += 1

        top_mean = top_sum / top_count
        top_var = (top_sum_sq / top_count) - (top_mean ** 2)
        top_std = torch.sqrt(top_var)

        bot_mean = bot_sum / bot_count
        bot_var = (bot_sum_sq / bot_count) - (bot_mean ** 2)
        bot_std = torch.sqrt(bot_var)

        return top_mean.item(), top_std.item(), bot_mean.item(), bot_std.item()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        # Convert to tensor if not already (assuming img is a PIL Image)
        if not isinstance(img, torch.Tensor):
            img = F.to_tensor(img)

        # Halve the image horizontally
        data_top = img[:, :img.shape[1]//2, :]
        data_bot = img[:, img.shape[1]//2:, :]
        
        data_top = self.transform_top(data_top)
        data_bot = self.transform_bot(data_bot)
        datareturn = [data_top, data_bot]
        
        # Missing Data
        if not "unimodal" in self.variant:
            datareturn[0] = apply_missing_mask(datareturn[0], self.zero_fill_masks[idx, 0])
            datareturn[1] = apply_missing_mask(datareturn[1], self.zero_fill_masks[idx, 1])
        
        return [
            datareturn, 
            label
        ]
