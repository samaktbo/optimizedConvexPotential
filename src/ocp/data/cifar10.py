from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Cifar10DataConfig:
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    download: bool = True
    seed: int = 0
    num_classes: int = 10  # keep classes {0, 1, ..., num_classes-1}
    augment: bool = True


def _require_torchvision():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "CIFAR-10 loader requires `torch` and `torchvision`. "
            "Install them, then re-run."
        ) from e


def _cifar10_transforms(*, train: bool, augment: bool):
    _require_torchvision()
    from torchvision import transforms

    # Standard CIFAR-10 normalization.
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    if train and augment:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def _filter_first_n_classes(dataset, num_classes: int):
    """Return a torch.utils.data.Subset keeping targets < num_classes."""
    if not (1 <= num_classes <= 10):
        raise ValueError(f"num_classes must be in [1, 10], got {num_classes}")

    _require_torchvision()
    from torch.utils.data import Subset

    # CIFAR10 stores labels in `targets` (list of ints).
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise AttributeError("Expected CIFAR10 dataset to have attribute `targets`.")

    idxs = [i for i, y in enumerate(targets) if int(y) < num_classes]
    return Subset(dataset, idxs)


def build_cifar10_datasets(cfg: Cifar10DataConfig):
    """Build (train_ds, test_ds), optionally filtered to first N classes."""
    _require_torchvision()
    from torchvision.datasets import CIFAR10

    train_ds = CIFAR10(
        root=cfg.data_dir,
        train=True,
        transform=_cifar10_transforms(train=True, augment=cfg.augment),
        download=cfg.download,
    )
    test_ds = CIFAR10(
        root=cfg.data_dir,
        train=False,
        transform=_cifar10_transforms(train=False, augment=False),
        download=cfg.download,
    )

    if cfg.num_classes != 10:
        train_ds = _filter_first_n_classes(train_ds, cfg.num_classes)
        test_ds = _filter_first_n_classes(test_ds, cfg.num_classes)

    return train_ds, test_ds


def _seed_worker(worker_id: int, base_seed: int) -> None:
    import random

    try:
        import numpy as np
    except ModuleNotFoundError:
        np = None

    seed = base_seed + worker_id
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)


def build_cifar10_loaders(cfg: Cifar10DataConfig):
    """Build (train_loader, test_loader) with deterministic-ish settings."""
    _require_torchvision()
    import torch
    from torch.utils.data import DataLoader

    train_ds, test_ds = build_cifar10_datasets(cfg)

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=g,
        worker_init_fn=(lambda wid: _seed_worker(wid, cfg.seed)),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=(lambda wid: _seed_worker(wid, cfg.seed)),
    )
    return train_loader, test_loader

