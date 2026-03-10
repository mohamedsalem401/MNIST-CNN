from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from mnist_hybrid.config import DataConfig


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset


def _maybe_subset(dataset: Dataset, max_samples: int, seed: int) -> Dataset:
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)


def build_datasets(config: DataConfig, seed: int) -> Dict[str, Dataset]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((config.normalize_mean,), (config.normalize_std,)),
        ]
    )

    train_full = datasets.MNIST(root=config.data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(root=config.data_dir, train=False, download=True, transform=transform)

    if config.train_size + config.val_size > len(train_full):
        raise ValueError(
            f"train_size + val_size exceeds MNIST train set: {config.train_size} + {config.val_size} > {len(train_full)}"
        )

    remaining = len(train_full) - config.train_size - config.val_size
    lengths = [config.train_size, config.val_size, remaining]
    generator = torch.Generator().manual_seed(seed)
    train, val, _ = random_split(train_full, lengths=lengths, generator=generator)

    train = _maybe_subset(train, config.max_train_samples, seed)
    val = _maybe_subset(val, config.max_val_samples, seed + 1)
    test = _maybe_subset(test, config.max_test_samples, seed + 2)

    return {
        "train": train,
        "val": val,
        "test": test,
    }


def build_dataloaders(config: DataConfig, seed: int) -> DataBundle:
    datasets_map = build_datasets(config, seed)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        datasets_map["train"],
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        datasets_map["val"],
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        datasets_map["test"],
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    return DataBundle(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        train_dataset=datasets_map["train"],
        val_dataset=datasets_map["val"],
        test_dataset=datasets_map["test"],
    )


def extract_labels(dataset: Dataset) -> torch.Tensor:
    if isinstance(dataset, Subset):
        base_labels = extract_labels(dataset.dataset)
        return base_labels[torch.tensor(dataset.indices, dtype=torch.long)]

    if hasattr(dataset, "targets"):
        labels = dataset.targets
        if isinstance(labels, list):
            return torch.tensor(labels, dtype=torch.long)
        return labels.clone().detach().long()

    raise TypeError(f"Cannot extract labels from dataset type: {type(dataset)}")
