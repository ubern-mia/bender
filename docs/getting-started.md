# Start here

Everything in this repository runs on a laptop CPU. Nothing needs a cluster, and the one
dataset used throughout downloads itself on first run.

## What you need

* **Python 3.13 or newer.**
* **[uv](https://github.com/astral-sh/uv)** for dependency management — `curl -LsSf https://astral.sh/uv/install.sh | sh`, or `brew install uv`.
* **[Graphviz](https://graphviz.org)** (optional) — only for the `torchview` computation-graph images in [episode 3](episodes/03-training-models.md).

## Set up

```bash
git clone https://github.com/ubern-mia/bender.git
cd bender

make setup      # create .venv with uv
make install    # install everything into .venv
```

The [`Makefile`](https://github.com/ubern-mia/bender/blob/main/Makefile) lives at the
repository root, next to a single shared `.venv/`, and every target is meant to be run from
there.

## Run something

=== "Train a model (episode 3)"

    ```bash
    make run-v1    # the naive baseline
    make run-v4    # Adam + TensorBoard
    make run-v7    # the version that beats the MedMNIST benchmark
    ```

    Each script also has a visualization mode that writes a layer summary, a computation
    graph and an ONNX export instead of training:

    ```bash
    make run-v4 MODE=visualize
    make netron FILE=training-models/results/v4/model.onnx
    ```

=== "Federate it (episode 9)"

    ```bash
    make run-fed-partition   # show how the data was split across five hospitals
    make run-fed             # train the federation (FedAvg, 5 clients, 10 rounds)
    make run-fed-local       # baseline 1: each hospital alone
    make run-fed-central     # baseline 2: the pooled upper bound
    ```

    Flags go through `FED_ARGS`:

    ```bash
    make run-fed FED_ARGS="--alpha 0.1 --strategy fedprox"
    ```

=== "Explore the data (episode 1)"

    The two exploration notebooks need no local setup at all — open them straight in
    Google Colab:

    * [`explore_dermamnist.ipynb`](https://colab.research.google.com/github/ubern-mia/bender/blob/main/exploratory-data-analysis/explore_dermamnist.ipynb)
    * [`explore_dicom.ipynb`](https://colab.research.google.com/github/ubern-mia/bender/blob/main/exploratory-data-analysis/explore_dicom.ipynb)

    To run them locally instead, add Jupyter to the environment first:

    ```bash
    uv pip install --python .venv/bin/python jupyterlab
    .venv/bin/jupyter lab exploratory-data-analysis/explore_dermamnist.ipynb
    ```

Cleaning up:

```bash
make clean       # remove training-models/results/ and __pycache__
make clean-venv  # remove .venv
```

## The dataset

Episodes 3 and 9 both use **DermaMNIST** from [MedMNIST v2](https://medmnist.com/) — 10,015
dermatoscopic images of skin lesions at 28×28 pixels, in seven classes, derived from the
[HAM10000](https://doi.org/10.1038/sdata.2018.161) collection. It is small enough to train
on in minutes and honest enough to be instructive: the classes are badly imbalanced
(melanocytic nevi alone are about two-thirds of the set), which is exactly the property
that makes accuracy a misleading metric. That lesson shows up again in
[episode 4](episodes/04-evaluation-and-deployment.md) and
[episode 9](episodes/09-federated-learning.md).

It downloads automatically the first time you run any script, into the `medmnist` cache in
your home directory.

!!! note "Why a toy dataset?"
    Because the point of the series is the *process*, not the score. 28×28 images let you
    run a full seven-version experiment sweep in an afternoon on a laptop, and every
    mistake the episodes warn about — class imbalance, overfitting, misleading averages,
    non-IID splits — is fully visible at that scale.

## Building this site locally

```bash
make docs-install   # adds mkdocs-material to .venv
make docs-serve     # live-reloading preview at http://127.0.0.1:8000
make docs-build     # render into site/
```

Pages live in [`docs/`](https://github.com/ubern-mia/bender/tree/main/docs). Pushing to
`main` publishes automatically via
[GitHub Actions](https://github.com/ubern-mia/bender/blob/main/.github/workflows/deploy-docs.yml).
