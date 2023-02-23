import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import pytorch_lightning as pl


class BaseTrainer:

    def __init__(self, model, dataset, scheduler_or_optimizer, **kwargs):
        self.model = model
        self.dataset = dataset
        self.scheduler_or_optimizer = scheduler_or_optimizer
        self.kwargs = kwargs
        self._lightning_trainer = pl.Trainer(**kwargs)


class TrainDevTrainer(BaseTrainer):

    def __init__(self, model, dataset, scheduler_or_optimizer, **kwargs):
        loaders_kwargs = {
            "batch_size": kwargs.get("batch_size", 32),
            "num_workers": kwargs.get("num_workers", 0),
            "random_state": kwargs.get("random_state", 42),
        }

        super().__init__(model, dataset, scheduler_or_optimizer, **kwargs)
        self.loaders = self.prepare_loaders(dataset, **loaders_kwargs)


    def prepare_loaders(self, dataset, **kwargs):
        random_state = kwargs.pop("random_state", None)
        generator = torch.Generator().manual_seed(random_state) if random_state else None
        replacement = kwargs.pop("replacement", False)
        num_samples = kwargs.pop("num_samples", None)


        loaders = {}
        for split in dataset.info.splits.keys():
            split_dataset = dataset.filter(lambda example: example["original_split"] == split)
            sampler = RandomSampler(split_dataset, replacement=replacement, num_samples=num_samples, generator=generator)
            loaders[split] = DataLoader(dataset, sampler=sampler, **kwargs)


class KFoldTrainer:
    pass


    





