from typing import Any
import os
import time

from tqdm import tqdm

import torch as th
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data

import medmnist
from medmnist import INFO
from torchvision import transforms

from sklearn.metrics import classification_report

# Define the torch.device you will use: use the cuda default.
device: th.device("cuda")


def _get_output_path() -> str:
    """Return an output path with the name of the current file."""
    base, _ = os.path.splitext(os.path.relpath(__file__))
    return base


def load_datasets():

    data_flag = "dermamnist"
    download = True

    info = INFO[data_flag]
    task = info["task"]
    n_channels = info["n_channels"]
    n_classes = len(info["label"])

    DataClass = getattr(medmnist, info["python_class"])

    transform_medmnist = transforms.Compose([transforms.ToTensor(), transforms.Pad(2)])

    training_transform_medmnist = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Pad(2),
            transforms.RandomCrop(
                size=(32, 32), padding=(0, 0, 5, 5), padding_mode="reflect"
            ),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    )

    # load the data
    data_train = DataClass(
        split="train", transform=training_transform_medmnist, download=download
    )
    data_test = DataClass(split="test", transform=transform_medmnist, download=download)

    return data_train, data_test


class CNN(nn.Module):
    def __init__(self):
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

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = th.reshape(x, (-1, 64))
        return self.classifier(x)


@th.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader):

    data_flag = "dermamnist"
    info = INFO[data_flag]

    # Evaluate the model with the given data loader.
    model.eval()
    correct = 0
    total = 0
    metrics = {}

    label_list = []
    pred_list = []

    for data in loader:
        images, labels = data[0], data[1]
        labels = labels.squeeze().long()

        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = th.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for x in predicted.numpy().tolist():
            pred_list.append(x)
        for x in labels.numpy().tolist():
            label_list.append(x)

    print(
        classification_report(
            label_list, pred_list, target_names=info["label"], digits=4
        )
    )

    # Return a dictionary of metrics. You should compute, at least, the accuracy.
    metrics["accuracy"] = correct / total
    return metrics


def main(
    output_path: str = None, batch_size: int = 8, num_epochs: int = 100, max_patience=30
):

    if output_path is None:
        output_path = _get_output_path()

    data_train, data_val = load_datasets()
    # Define the PyTorch data loaders for the training and test datasets.
    # Use the given batch_size and remember that the training loader should
    # shuffle the batches each epoch.
    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(data_val, batch_size=batch_size, shuffle=False)

    # Define the model and move it to the device. Define the optimizer for
    # the parameters of the model.
    model = CNN()
    optimizer = th.optim.Adam(model.parameters(), lr=0.005)
    loss_function = th.nn.CrossEntropyLoss()

    # Compute the number of parameters of the model
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    # Create the tensorboard summary writer.
    summary = SummaryWriter(output_path, purge_step=0)

    # Log the number of parameters to tensorboard
    summary.add_scalar("num_params", num_params)

    # Iteration counter
    it = 0

    # Patience starts with the given maximum patience. Patience should decrease
    # every time the model is evaluated and the performance did not improve.
    patience = max_patience
    # We need to keep track of the best performance reached so far.
    best_accuracy = 0

    # Number of iterations required in one epoch
    epoch_length = len(loader_train)

    # Repeat training the given number of epochs
    for epoch in range(num_epochs):

        print(f"Starting epoch {epoch + 1}/{num_epochs}...")

        # Run one epoch
        for batch in tqdm(loader_train):

            it += 1

            # IMPORTANT NOTE: REMEMBER TO SET THE TRAINING STATE OF THE MODEL.
            # Call .train() before training and .eval() before evaluation every
            # time!!
            model.train()
            inputs = batch[0]
            labels = batch[1]
            labels = labels.squeeze().long()

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_function(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Log the training loss once every 50 iterations
            if (it % 50) == 0:
                # Log the loss to tensorboard (using summary.add_scalar)
                summary.add_scalar("loss", loss, it)

            # Run validation, update patience, and save the model once every epoch.
            # You could put this code outside the inner training loop, but
            # doing it here allows you to run validation more than once per epoch.
            if (it % epoch_length) == 0:
                metrics = evaluate_model(model, loader_val)

                # Loop over the metrics and log them to tensorboard
                for key in metrics.keys():
                    summary.add_scalar(key, metrics[key], it)

                accuracy = metrics["accuracy"]
                if accuracy > best_accuracy:
                    # Update patience and best_accuracy
                    patience = max_patience
                    best_accuracy = accuracy
                    model_file = os.path.join(output_path, "best_model.pt")
                    # Save the model to the given `model_file`.
                    # In principle, you should save not only the model,
                    # but also the optimizer just in case you want to resume an
                    # interrupted training.
                    th.save(model.state_dict(), model_file)
                else:
                    patience -= 1

                print(f"My remaining patience is {patience}.")
                print(f"Current accuracy is {accuracy}, and best is: {best_accuracy}.")

                if patience == 0:
                    print("My patience ran out.")
                    return


if __name__ == "__main__":
    train = True

    if train:
        main()
    else:
        # Test model loading after training.
        model = CNN()
        model.load_state_dict(th.load(_get_output_path() + "/best_model.pt"))
        model.eval()

        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        num_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {num_params}")

        data_train, data_val = load_datasets()
        loader_val = DataLoader(data_val, batch_size=8, shuffle=False)

        metrics = evaluate_model(model, loader_val)
        print(f"Best accuracy is: ", metrics["accuracy"])
