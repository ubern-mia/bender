"""
dermamnist_v3_lr0p005_val_patience:
changelog: updated learning rate to 0.005 from 0.000005 and included validation patience.
"""

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from shared.data import load_datasets
from shared.model import get_output_path
from shared.utils import train, visualize_model, test

VERSION = "v3"


class CNN(nn.Module):
    """
    A simple 4 layered CNN to run classification on dermamnist.
    """

    def __init__(self):
        """
        Definition of layers in the CNN.
        """

        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, (5, 5), padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # (32, 32, 32)
            nn.Conv2d(64, 64, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # (32, 32, 32)
            nn.Conv2d(64, 64, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # (32, 32, 32)
            nn.Conv2d(64, 64, (3, 3), padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # (64, 16, 16)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(nn.Linear(64, 7))

    def forward(self, in_tensor):
        """
        Forward pass through the CNN.
        """

        in_tensor = self.features(in_tensor)
        in_tensor = self.avgpool(in_tensor)
        in_tensor = torch.reshape(in_tensor, (-1, 64))
        return self.classifier(in_tensor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dermamnist_v3_lr0p005_val_patience")
    parser.add_argument(
        "mode",
        nargs="?",
        default="visualize",
        choices=["train", "test", "visualize"],
        help="Operation to perform (default: visualize)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--max-patience", type=int, default=30)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="[test] path to a checkpoint file (default: best_model.pt in output dir)",
    )
    args = parser.parse_args()

    output_path = get_output_path(VERSION)

    if args.mode == "train":
        data_train, data_val = load_datasets("train")
        loader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
        loader_val = DataLoader(data_val, batch_size=args.batch_size, shuffle=False)
        model = CNN()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
        train(
            model, optimizer, loader_train, loader_val, output_path,
            num_epochs=args.num_epochs,
            max_patience=args.max_patience,
            use_tensorboard=False,
        )
    elif args.mode == "test":
        test(CNN, output_path, args.batch_size, checkpoint_path=args.checkpoint)
    elif args.mode == "visualize":
        visualize_model(CNN(), output_path)
