"""
shared/data.py: dermamnist data loading utilities shared across all versions.
"""

import medmnist
from medmnist import INFO
from torchvision import transforms


def load_datasets(flag, training_transform=None):
    """
    Load dermamnist datasets.

    flag: 'train' returns (train_data, val_data), 'test' returns (train_data, test_data).
    training_transform: optional transform override for the train split (e.g. augmentation).
    """
    data_flag = "dermamnist"
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])

    base_transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2)])
    train_transform = training_transform if training_transform is not None else base_transform

    data_train = DataClass(split="train", transform=train_transform, download=True)

    if flag == "train":
        data_next = DataClass(split="val", transform=base_transform, download=True)
    elif flag == "test":
        data_next = DataClass(split="test", transform=base_transform, download=True)

    return data_train, data_next
