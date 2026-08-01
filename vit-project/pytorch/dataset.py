"""CIFAR-10 loaders: plain (no augmentation) and augmented variants."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_transforms(augment: bool):
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.25),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, test_tf


def get_dataloaders(
    data_dir="../../data",
    batch_size=128,
    augment=False,
    num_workers=2,
    subset_size=None,
):
    """subset_size: if set, restrict the train set to this many images (for quick smoke tests)."""
    train_tf, test_tf = get_transforms(augment)

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    if subset_size is not None:
        train_set = Subset(train_set, list(range(subset_size)))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders(augment=False, subset_size=256)
    images, labels = next(iter(train_loader))
    print(f"batch images: {tuple(images.shape)}, labels: {tuple(labels.shape)}")
    print(f"train batches: {len(train_loader)}, test batches: {len(test_loader)}")
