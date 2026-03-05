from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from mnist_nn.model import MNISTMLP

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MLP on MNIST.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/mnist_mlp.pt"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_subset(dataset, max_samples: int, seed: int):
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)


def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    max_train_samples: int,
    max_test_samples: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )

    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_dataset = maybe_subset(train_dataset, max_train_samples, seed)
    test_dataset = maybe_subset(test_dataset, max_test_samples, seed)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, test_loader


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_one_epoch(
    model: MNISTMLP,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits, labels)

    avg_loss = running_loss / len(loader)
    avg_acc = running_acc / len(loader)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(
    model: MNISTMLP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits, labels)

    avg_loss = running_loss / len(loader)
    avg_acc = running_acc / len(loader)
    return avg_loss, avg_acc


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
    )

    model = MNISTMLP(hidden_sizes=args.hidden_sizes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc * 100:.2f}% | "
            f"test_loss={test_loss:.4f} test_acc={test_acc * 100:.2f}%"
        )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "hidden_sizes": list(args.hidden_sizes),
        "best_test_accuracy": best_acc,
        "best_epoch": best_epoch,
        "normalize_mean": MNIST_MEAN,
        "normalize_std": MNIST_STD,
    }
    torch.save(checkpoint, args.checkpoint)

    print(
        f"Saved checkpoint to {args.checkpoint} | "
        f"best_test_accuracy={best_acc * 100:.2f}% (epoch {best_epoch})"
    )


if __name__ == "__main__":
    main()
