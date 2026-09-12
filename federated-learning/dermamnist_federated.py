"""
dermamnist_federated:
a simulated federation over dermamnist, using Flower.

The point of this script is not to win a benchmark. It is to let you watch the
three numbers that matter sit next to each other:

  local       what one hospital gets training alone on its own shard
  federated   what the same hospitals get by training together, data in place
  centralized what they would get if the data could legally be pooled

Federated learning is worth the trouble when `federated` lands closer to
`centralized` than to `local`. Run all three modes and see for yourself.

The data is split across clients with a Dirichlet distribution, which is the
usual way to fake the label skew you get for free in real life: a tertiary
referral centre sees the rare classes, a district hospital sees the common ones.
Lower --alpha means more skew.
"""

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

# dermamnist loading lives with episode 3, so that both episodes see exactly the
# same data and the accuracies below are comparable with the ones over there.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training-models"))
from shared.data import load_datasets  # noqa: E402

from flwr.client import ClientApp, NumPyClient  # noqa: E402
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays  # noqa: E402
from flwr.server import ServerApp, ServerAppComponents, ServerConfig  # noqa: E402
from flwr.server.strategy import FedAvg, FedProx  # noqa: E402
from flwr.simulation import run_simulation  # noqa: E402

VERSION = "federated"
NUM_CLASSES = 7
OUTPUT_DIR = Path(__file__).resolve().parent / "dermamnist_federated"


class CNN(nn.Module):
    """
    The same 4 layered CNN as dermamnist_v4, so that the only thing that
    changes between episode 3 and this episode is *where the data lives*.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, (5, 5), padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Linear(64, NUM_CLASSES))

    def forward(self, in_tensor):
        in_tensor = self.features(in_tensor)
        in_tensor = self.avgpool(in_tensor)
        in_tensor = torch.reshape(in_tensor, (-1, 64))
        return self.classifier(in_tensor)


# ── Splitting one dataset into many hospitals ─────────────────────────────────


def dirichlet_partition(labels, num_clients: int, alpha: float, seed: int = 42):
    """
    Split sample indices across clients so that every client gets a different
    class mixture. alpha -> infinity approaches an even (IID) split; alpha -> 0
    gives each client almost a single class.
    """
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(num_clients)]

    for class_id in range(NUM_CLASSES):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        cuts = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        for client_id, shard in enumerate(np.split(class_indices, cuts)):
            client_indices[client_id].extend(shard.tolist())

    return [sorted(indices) for indices in client_indices]


def describe_partition(client_indices, labels, class_names):
    """Print the per-client class histogram. Look at this before you train."""
    header = f"{'client':>8} {'n':>6}  " + "  ".join(f"{name[:10]:>10}" for name in class_names)
    print("\n" + header)
    print("-" * len(header))
    for client_id, indices in enumerate(client_indices):
        counts = np.bincount(labels[indices], minlength=NUM_CLASSES)
        row = f"{client_id:>8} {len(indices):>6}  " + "  ".join(f"{c:>10}" for c in counts)
        print(row)
    print()


# ── Plain torch training and evaluation, used by every mode ───────────────────


def train_locally(model, loader, epochs, learning_rate=0.005, global_params=None, proximal_mu=0.0):
    """
    Train in place for a few epochs. When proximal_mu > 0 this is the FedProx
    update: an extra penalty keeps the local model from drifting too far from
    the global one, which is what stops skewed clients pulling in circles.
    """
    model.train()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        for images, labels in loader:
            labels = labels.squeeze(-1).long()
            optimizer.zero_grad()
            loss = loss_function(model(images), labels)

            if proximal_mu > 0.0 and global_params is not None:
                proximal_term = sum(
                    torch.square(torch.norm(local - reference))
                    for local, reference in zip(model.parameters(), global_params)
                )
                loss = loss + (proximal_mu / 2) * proximal_term

            loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate(model, loader):
    """Return (average loss, accuracy, balanced accuracy) on loader."""
    model.eval()
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, total = 0.0, 0, 0
    per_class_correct = np.zeros(NUM_CLASSES)
    per_class_total = np.zeros(NUM_CLASSES)

    for images, labels in loader:
        labels = labels.squeeze(-1).long()
        outputs = model(images)
        total_loss += loss_function(outputs, labels).item()
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        for class_id in range(NUM_CLASSES):
            mask = labels == class_id
            per_class_total[class_id] += mask.sum().item()
            per_class_correct[class_id] += (predicted[mask] == class_id).sum().item()

    seen = per_class_total > 0
    balanced = float(np.mean(per_class_correct[seen] / per_class_total[seen]))
    return total_loss / total, correct / total, balanced


# ── Parameter exchange, and the FedBN variant ─────────────────────────────────


def shared_keys(model, fedbn: bool):
    """
    Which tensors actually travel over the wire.

    With --fedbn, every BatchNorm parameter and running statistic stays at the
    hospital that computed it. Those statistics encode the scanner, so keeping
    them local absorbs a surprising amount of the site-to-site shift.
    """
    keys = list(model.state_dict().keys())
    if not fedbn:
        return keys
    bn_modules = {name for name, module in model.named_modules() if isinstance(module, nn.BatchNorm2d)}
    return [key for key in keys if key.rsplit(".", 1)[0] not in bn_modules]


def get_parameters(model, keys):
    state = model.state_dict()
    return [state[key].cpu().numpy() for key in keys]


def set_parameters(model, keys, parameters):
    state = OrderedDict(
        (key, torch.tensor(value)) for key, value in zip(keys, parameters)
    )
    model.load_state_dict(state, strict=False)


# ── The Flower client: episode 3's training loop, wrapped ─────────────────────


class HospitalClient(NumPyClient):
    """One hospital. It never sees another hospital's data, only its weights."""

    def __init__(self, train_subset, test_dataset, args):
        self.args = args
        self.model = CNN()
        self.keys = shared_keys(self.model, args.fedbn)
        self.loader_train = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        self.loader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        self.num_train = len(train_subset)

    def fit(self, parameters, config):
        set_parameters(self.model, self.keys, parameters)
        proximal_mu = float(config.get("proximal_mu", 0.0))
        reference = [p.detach().clone() for p in self.model.parameters()] if proximal_mu else None
        train_locally(
            self.model,
            self.loader_train,
            epochs=self.args.local_epochs,
            learning_rate=self.args.learning_rate,
            global_params=reference,
            proximal_mu=proximal_mu,
        )
        return get_parameters(self.model, self.keys), self.num_train, {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, self.keys, parameters)
        loss, accuracy, balanced = evaluate(self.model, self.loader_test)
        return loss, len(self.loader_test.dataset), {"accuracy": accuracy, "balanced_accuracy": balanced}


# ── The three modes ───────────────────────────────────────────────────────────


def run_federated(args, client_subsets, data_test):
    """Train across the federation with FedAvg (or FedProx), round by round."""
    template = CNN()
    keys = shared_keys(template, args.fedbn)
    initial_parameters = ndarrays_to_parameters(get_parameters(template, keys))

    loader_test = DataLoader(data_test, batch_size=args.batch_size, shuffle=False)
    history = []

    def client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        return HospitalClient(client_subsets[partition_id], data_test, args).to_client()

    def central_evaluate(server_round, parameters, config):
        """
        The server holds out a test set and scores the global model on it.

        Not available under --fedbn: the BatchNorm parameters never reach the
        server, so there is no complete global model to score. That is not a
        bug in the code, it is the actual trade-off FedBN makes, and the reason
        we fall back to asking each client to evaluate locally instead.
        """
        model = CNN()
        set_parameters(model, keys, parameters)
        loss, accuracy, balanced = evaluate(model, loader_test)
        history.append((server_round, accuracy, balanced))
        print(
            f"[round {server_round:>3}] central test accuracy {accuracy:.4f}  "
            f"balanced {balanced:.4f}"
        )
        return loss, {"accuracy": accuracy, "balanced_accuracy": balanced}

    def weighted_average(metrics):
        total = sum(num for num, _ in metrics)
        return {
            key: sum(num * m[key] for num, m in metrics) / total
            for key in metrics[0][1]
        }

    def distributed_report(server_round, metrics):
        print(
            f"[round {server_round:>3}] federated test accuracy {metrics['accuracy']:.4f}  "
            f"balanced {metrics['balanced_accuracy']:.4f}"
        )
        history.append((server_round, metrics["accuracy"], metrics["balanced_accuracy"]))
        return metrics

    def server_fn(context: Context):
        common = dict(
            fraction_fit=1.0,
            min_fit_clients=args.num_clients,
            min_available_clients=args.num_clients,
            initial_parameters=initial_parameters,
        )
        if args.fedbn:
            common.update(
                fraction_evaluate=1.0,
                min_evaluate_clients=args.num_clients,
                evaluate_metrics_aggregation_fn=lambda m: distributed_report(
                    len(history) + 1, weighted_average(m)
                ),
            )
        else:
            common.update(fraction_evaluate=0.0, evaluate_fn=central_evaluate)

        if args.strategy == "fedprox":
            strategy = FedProx(proximal_mu=args.proximal_mu, **common)
        else:
            strategy = FedAvg(**common)

        return ServerAppComponents(
            strategy=strategy, config=ServerConfig(num_rounds=args.num_rounds)
        )

    run_simulation(
        server_app=ServerApp(server_fn=server_fn),
        client_app=ClientApp(client_fn=client_fn),
        num_supernodes=args.num_clients,
        backend_config={"client_resources": {"num_cpus": args.cpus_per_client, "num_gpus": 0.0}},
    )

    if history:
        rounds = [r for r, _, _ in history]
        plt.figure()
        plt.plot(rounds, [a for _, a, _ in history], label="accuracy")
        plt.plot(rounds, [b for _, _, b in history], label="balanced accuracy")
        plt.xlabel("Federated round")
        plt.ylabel("Test accuracy")
        plt.title(f"{args.strategy}, {args.num_clients} clients, alpha={args.alpha}")
        plt.grid()
        plt.legend()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plot_path = OUTPUT_DIR / f"{args.strategy}_alpha{args.alpha}_clients{args.num_clients}.png"
        plt.savefig(plot_path)
        print(f"\nSaved {plot_path}")
        print(f"Final federated accuracy: {history[-1][1]:.4f}")


def run_local(args, client_subsets, data_test):
    """Baseline 1: each hospital trains alone, and is scored on the same test set."""
    loader_test = DataLoader(data_test, batch_size=args.batch_size, shuffle=False)
    accuracies, balanced_accuracies = [], []

    for client_id, subset in enumerate(client_subsets):
        model = CNN()
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True)
        train_locally(model, loader, epochs=args.num_rounds * args.local_epochs, learning_rate=args.learning_rate)
        _, accuracy, balanced = evaluate(model, loader_test)
        accuracies.append(accuracy)
        balanced_accuracies.append(balanced)
        print(
            f"client {client_id} (n={len(subset):>5}) alone: accuracy {accuracy:.4f}  "
            f"balanced {balanced:.4f}"
        )

    print(
        f"\nMean local-only accuracy: {np.mean(accuracies):.4f}  (best {max(accuracies):.4f})"
        f"\nMean local-only balanced accuracy: {np.mean(balanced_accuracies):.4f}"
        f"  (best {max(balanced_accuracies):.4f})  <- compare this one"
    )


def run_centralized(args, data_train, data_test):
    """Baseline 2: the upper bound you would get if the data could be pooled."""
    model = CNN()
    loader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    loader_test = DataLoader(data_test, batch_size=args.batch_size, shuffle=False)
    train_locally(model, loader_train, epochs=args.num_rounds * args.local_epochs, learning_rate=args.learning_rate)
    _, accuracy, balanced = evaluate(model, loader_test)
    print(f"\nCentralized (pooled) accuracy: {accuracy:.4f}  balanced {balanced:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dermamnist_federated")
    parser.add_argument(
        "mode",
        nargs="?",
        default="federated",
        choices=["federated", "local", "centralized", "partition"],
        help="What to run (default: federated). 'partition' only prints the data split.",
    )
    parser.add_argument("--num-clients", type=int, default=5, help="number of simulated hospitals")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration; lower is more skewed")
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--strategy", choices=["fedavg", "fedprox"], default="fedavg")
    parser.add_argument("--proximal-mu", type=float, default=0.1, help="[fedprox] strength of the proximal term")
    parser.add_argument("--fedbn", action="store_true", help="keep BatchNorm parameters local (FedBN)")
    parser.add_argument("--cpus-per-client", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    from medmnist import INFO

    class_names = list(INFO["dermamnist"]["label"].values())

    data_train, data_test = load_datasets("test")
    labels = np.array(data_train.labels).squeeze()

    client_indices = dirichlet_partition(labels, args.num_clients, args.alpha, seed=args.seed)
    describe_partition(client_indices, labels, class_names)

    client_subsets = [Subset(data_train, indices) for indices in client_indices]

    if args.mode == "partition":
        pass
    elif args.mode == "federated":
        run_federated(args, client_subsets, data_test)
    elif args.mode == "local":
        run_local(args, client_subsets, data_test)
    elif args.mode == "centralized":
        run_centralized(args, data_train, data_test)
