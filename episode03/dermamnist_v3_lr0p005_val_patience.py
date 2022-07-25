"""
dermamnist_v3_lr0p005_val_patience:
changelog: updated learning rate to 0.005 from 0.000005 and included validation patience.
"""

import os

import medmnist
from medmnist import INFO

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

# Define the torch.device you will use.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_output_path() -> str:
    """
    Return an output path with the name of the current file.
    """

    base, _ = os.path.splitext(os.path.relpath(__file__))
    return base


def load_datasets(flag):
    """
    load_datasets loads the dermamnist data.
    'flag' takes two options:
        'train': loads the training set as first output, validation set as second.
        'test' : loads the training set as first output, test set as second.
    """

    data_flag = "dermamnist"
    download = True
    info = INFO[data_flag]

    DataClass = getattr(medmnist, info["python_class"])

    transform_medmnist = transforms.Compose([transforms.ToTensor(), transforms.Pad(2)])

    data_train = DataClass(
        split="train", transform=transform_medmnist, download=download
    )
    if flag == "train":
        data_next = DataClass(
            split="val", transform=transform_medmnist, download=download
        )
    elif flag == "test":
        data_next = DataClass(
            split="test", transform=transform_medmnist, download=download
        )

    return data_train, data_next


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


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader):
    """
    Evaluate model while training.
    """

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
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for pred in predicted.numpy().tolist():
            pred_list.append(pred)
        for label in labels.numpy().tolist():
            label_list.append(label)

    print(
        classification_report(
            label_list, pred_list, target_names=list(info["label"].values()), digits=4
        )
    )

    metrics["accuracy"] = correct / total
    return metrics


def train(output_path: str = None, batch_size: int = 8, num_epochs: int = 100, max_patience=30):
    """
    Train model.
    """

    if output_path is None:
        output_path = _get_output_path()
        os.makedirs(output_path, exist_ok=True)

    data_train, data_val = load_datasets("train")
    # Define the PyTorch data loaders for the training and test datasets.
    # Use the given batch_size and remember that the training loader should
    # shuffle the batches each epoch.
    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(data_val, batch_size=batch_size, shuffle=False)

    # Define the model and move it to the device. Define the optimizer for
    # the parameters of the model.
    model = CNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    loss_function = torch.nn.CrossEntropyLoss()

    # Compute the number of parameters of the model
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    # Iteration counter
    iteration = 0

    train_loss = []
    val_acc = []

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

            iteration += 1

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
            if (iteration % 50) == 0:
                train_loss.append([loss, iteration])

            # Run validation, update patience, and save the model once every epoch.
            # You could put this code outside the inner training loop, but
            # doing it here allows you to run validation more than once per epoch.
            if (iteration % epoch_length) == 0:

                batch = next(iter(loader_val))

                inputs = batch[0]
                labels = batch[1]
                labels = labels.squeeze().long()

                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                outputs = model(inputs)

                # Loop over the metrics for validation, loss and accuracy.
                metrics = evaluate_model(model, loader_val)

                # Loop over the metrics and log them to tensorboard
                for key in metrics.keys():
                    val_acc.append([metrics[key], iteration])

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
                    torch.save(model.state_dict(), model_file)
                else:
                    patience -= 1

                print(f"My remaining patience is {patience}.")
                print(f"Current accuracy is {accuracy}, and best is: {best_accuracy}.")

                if patience == 0:
                    print("My validation patience ran out.")
                    break
        if patience == 0:
            break

    plt.figure()
    plt.plot(
        [it for loss, it in train_loss],
        [loss.detach().item() for loss, it in train_loss],
    )
    plt.xlabel("Iterations")
    plt.ylabel("Training loss")
    plt.grid()
    plt.savefig(os.path.join(output_path, "train_loss.png"))

    plt.figure()
    plt.plot([it for acc, it in val_acc], [acc for acc, it in val_acc])
    plt.xlabel("Iterations")
    plt.ylabel("Validation accuracy")
    plt.grid()
    plt.savefig(os.path.join(output_path, "val_acc.png"))


def test():
    """
    Test model after training.
    """

    model = CNN()
    model_path = _get_output_path()
    model.load_state_dict(torch.load(model_path + "/best_model.pt"))

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    _, data_test = load_datasets("test")
    loader_test = DataLoader(data_test, batch_size=8, shuffle=False)

    metrics = evaluate_model(model, loader_test)
    print(f"Test accuracy is: {metrics['accuracy']}")


if __name__ == "__main__":

    TRAINING_MODE = True
    if TRAINING_MODE:
        train()
    else:
        test()
