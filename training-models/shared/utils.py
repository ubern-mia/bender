"""
shared/utils.py: training, evaluation, visualization, and testing utilities
shared across all dermamnist versions.
"""

import glob
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import classification_report
from medmnist import INFO
from torchinfo import summary as torchinfo_summary
from torchview import draw_graph

from shared.data import load_datasets


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader) -> dict:
    """
    Evaluate model on loader; prints a classification report and returns
    a dict with 'accuracy'.
    """
    data_flag = "dermamnist"
    info = INFO[data_flag]

    model.eval()
    correct = 0
    total = 0
    label_list = []
    pred_list = []

    for data in loader:
        images, labels = data[0], data[1]
        labels = labels.squeeze().long()

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pred_list.extend(predicted.numpy().tolist())
        label_list.extend(labels.numpy().tolist())

    print(
        classification_report(
            label_list, pred_list, target_names=list(info["label"].values()), digits=4
        )
    )
    return {"accuracy": correct / total}


def _latest_checkpoint(output_path: str):
    """Return the path of the most recent epoch checkpoint, or None."""
    pattern = os.path.join(output_path, "checkpoint_epoch_*.pt")
    checkpoints = sorted(glob.glob(pattern))
    return checkpoints[-1] if checkpoints else None


def _save_checkpoint(
    output_path: str,
    epoch: int,
    iteration: int,
    model: nn.Module,
    optimizer,
    best_accuracy: float,
    patience,
):
    """Persist a full training checkpoint after every validation pass."""
    path = os.path.join(output_path, f"checkpoint_epoch_{epoch + 1:04d}.pt")
    torch.save(
        {
            "epoch": epoch,
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_accuracy": best_accuracy,
            "patience": patience,
        },
        path,
    )
    return path


def train(
    model: nn.Module,
    optimizer,
    loader_train: DataLoader,
    loader_val: DataLoader,
    output_path: str,
    num_epochs: int = 100,
    max_patience: int = None,
    use_tensorboard: bool = False,
):
    """
    Train model, saving a checkpoint and updating best_model.pt after every
    validation pass.

    Automatically resumes from the latest checkpoint in output_path when one
    is found — no extra flags needed.

    max_patience: if set, training stops early when val accuracy stops improving.
    use_tensorboard: if True logs to TensorBoard; otherwise saves matplotlib plots.
    """
    os.makedirs(output_path, exist_ok=True)

    loss_function = torch.nn.CrossEntropyLoss()
    epoch_length = len(loader_train)

    # ── Resume state ──────────────────────────────────────────────────────────
    start_epoch = 0
    iteration = 0
    patience = max_patience
    best_accuracy = 0.0

    latest = _latest_checkpoint(output_path)
    if latest:
        print(f"Resuming from {latest}")
        saved = torch.load(latest)
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_epoch = saved["epoch"] + 1
        iteration = saved["iteration"]
        best_accuracy = saved["best_accuracy"]
        if max_patience is not None:
            patience = saved.get("patience", max_patience)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    if start_epoch >= num_epochs:
        print(f"Already trained for {num_epochs} epochs — nothing to do.")
        return

    # ── Logging setup ─────────────────────────────────────────────────────────
    if use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        # purge_step keeps existing TB events intact up to where we left off
        summary = SummaryWriter(output_path, purge_step=iteration)
        if start_epoch == 0:
            summary.add_scalar("num_params", num_params)
    else:
        train_loss = []
        val_acc = []

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs):

        print(f"Starting epoch {epoch + 1}/{num_epochs}...")

        for batch in tqdm(loader_train):
            iteration += 1

            model.train()
            inputs, labels = batch[0], batch[1].squeeze().long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            if (iteration % 50) == 0:
                if use_tensorboard:
                    summary.add_scalar("loss/train", loss, iteration)
                else:
                    train_loss.append((loss.detach().item(), iteration))

            if (iteration % epoch_length) == 0:
                # ── Validation ────────────────────────────────────────────────
                if use_tensorboard:
                    train_metrics = evaluate_model(model, loader_train)
                    for key, val in train_metrics.items():
                        summary.add_scalar(f"{key}/train", val, iteration)

                    val_batch = next(iter(loader_val))
                    val_inputs = val_batch[0]
                    val_labels = val_batch[1].squeeze().long()
                    optimizer.zero_grad()
                    val_outputs = model(val_inputs)
                    val_loss = loss_function(val_outputs, val_labels)
                    summary.add_scalar("loss/val", val_loss, iteration)

                val_metrics = evaluate_model(model, loader_val)
                accuracy = val_metrics["accuracy"]

                if use_tensorboard:
                    for key, val in val_metrics.items():
                        summary.add_scalar(f"{key}/val", val, iteration)
                else:
                    val_acc.append((accuracy, iteration))

                # ── Patience + best model ─────────────────────────────────────
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    if max_patience is not None:
                        patience = max_patience
                    torch.save(
                        model.state_dict(),
                        os.path.join(output_path, "best_model.pt"),
                    )
                elif max_patience is not None:
                    patience -= 1

                # ── Save full checkpoint every validation pass ─────────────────
                ckpt_path = _save_checkpoint(
                    output_path, epoch, iteration, model, optimizer,
                    best_accuracy, patience,
                )
                print(f"Checkpoint saved → {ckpt_path}")

                if max_patience is not None:
                    print(f"My remaining patience is {patience}.")
                print(f"Current accuracy is {accuracy:.4f}, best is {best_accuracy:.4f}.")

                if max_patience is not None and patience == 0:
                    print("My validation patience ran out.")
                    return

    # ── Final matplotlib plots (non-TensorBoard runs) ─────────────────────────
    if not use_tensorboard:
        plt.figure()
        plt.plot([it for _, it in train_loss], [loss for loss, _ in train_loss])
        plt.xlabel("Iterations")
        plt.ylabel("Training loss")
        plt.grid()
        plt.savefig(os.path.join(output_path, "train_loss.png"))

        plt.figure()
        plt.plot([it for _, it in val_acc], [acc for acc, _ in val_acc])
        plt.xlabel("Iterations")
        plt.ylabel("Validation accuracy")
        plt.grid()
        plt.savefig(os.path.join(output_path, "val_acc.png"))


def visualize_model(model: nn.Module, output_path: str):
    """
    Save model summary (torchinfo), computation graph (torchview), and ONNX
    export (for netron) to output_path.
    """
    os.makedirs(output_path, exist_ok=True)

    model_stats = torchinfo_summary(model, input_size=(1, 3, 32, 32), verbose=0)
    with open(os.path.join(output_path, "model_summary.txt"), "w") as f:
        f.write(str(model_stats))
    print(f"Model summary saved to {output_path}/model_summary.txt")

    graph = draw_graph(
        model,
        input_size=(1, 3, 32, 32),
        save_graph=True,
        filename="model_graph",
        directory=output_path,
        expand_nested=True,
    )
    graph.visual_graph.render(format="png", cleanup=True)
    print(f"Model graph saved to {output_path}/model_graph.png")

    dummy_input = torch.randn(1, 3, 32, 32)
    onnx_path = os.path.join(output_path, "model.onnx")
    try:
        torch.onnx.export(model, dummy_input, onnx_path, opset_version=11)
        print(f"ONNX model saved. Run: netron {onnx_path}")
    except ModuleNotFoundError as e:
        print(f"ONNX export skipped ({e}). Install the missing package with: pip install onnxscript")


def test(
    model_class,
    output_path: str,
    batch_size: int = 8,
    checkpoint_path: str = None,
):
    """
    Instantiate model_class, load a checkpoint, and evaluate on the test set.

    checkpoint_path: explicit path to any checkpoint file (epoch checkpoint or
        best_model.pt). Defaults to best_model.pt in output_path. Pass the
        path to a checkpoint_epoch_NNNN.pt to evaluate a mid-training snapshot.
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join(output_path, "best_model.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. "
            "Run training first, or specify a checkpoint with --checkpoint."
        )

    model = model_class()
    saved = torch.load(checkpoint_path)

    # Support both the epoch-checkpoint dict format and the bare state-dict
    # format written by older versions of best_model.pt.
    if isinstance(saved, dict) and "model_state_dict" in saved:
        model.load_state_dict(saved["model_state_dict"])
        epoch_label = saved.get("epoch", "?")
        if epoch_label != "?":
            epoch_label += 1  # stored as 0-indexed
        best_acc = saved.get("best_accuracy", "?")
        print(f"Loaded checkpoint: epoch {epoch_label}, best accuracy so far {best_acc:.4f}")
    else:
        model.load_state_dict(saved)
        print(f"Loaded model weights from {checkpoint_path}")

    print("\nModel's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nNumber of parameters: {num_params}")

    _, data_test = load_datasets("test")
    loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=False)

    metrics = evaluate_model(model, loader_test)
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
