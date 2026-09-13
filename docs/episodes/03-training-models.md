# Episode 3 — Good Model Training Shall You Strive For

<div class="video-embed">
  <iframe src="https://www.youtube-nocookie.com/embed/f0wd8EvRiH0" title="BENDER Episode 3: Good Model Training Shall You Strive For" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

<div class="episode-meta">
  <span><strong>Runtime</strong> 8:27</span>
  <span><strong>Published</strong> 18 August 2022</span>
  <span><strong>Characters</strong> Satish, Mike, Carmen</span>
  <span><strong>Companion</strong> 7 scripts · 2 notebooks · <code>make run-v1</code>…<code>run-v7</code></span>
</div>

## What happens in the episode

Satish sets out to train his model, and Mike and Carmen talk him through doing it *in an
order that lets him learn something*. The episode is about experimental discipline rather
than architecture: change one thing, log everything, compare against what you had before,
and keep the whole trail so that three months later you can still say why version 4 exists.

The companion code takes that literally. There are seven versions of the same DermaMNIST
classifier, and the diff between consecutive versions is usually a single line.

## The main idea: one change at a time

This is the habit the episode is built around, and it is worth stating plainly.

When a training run disappoints, the temptation is to change the learning rate *and* the
architecture *and* add augmentation, then re-run. If it improves you have learned nothing
about which change did it; if it gets worse you have learned less. Changing one thing per
version is slower per experiment and far faster per project, because every result is
attributable.

The seven versions here are exactly that discipline, applied honestly — including the two
versions that made things *worse*, which are kept rather than quietly deleted.

## The seven versions

| Version | The one change | Test accuracy | Δ |
|---|---|---|---|
| [v1](https://github.com/ubern-mia/bender/blob/main/training-models/dermamnist_v1_initial.py) | Baseline 4-layer CNN, SGD, lr = 5e-6, momentum = 0.5 | ~0.65 | — |
| [v2](https://github.com/ubern-mia/bender/blob/main/training-models/dermamnist_v2_momentum0p9.py) | Momentum 0.5 → 0.9 | ~0.65 | ≈ 0.00 |
| [v3](https://github.com/ubern-mia/bender/blob/main/training-models/dermamnist_v3_lr0p005_val_patience.py) | Learning rate 5e-6 → 0.005, plus validation patience | ~0.75 | **+0.10** |
| [v4](https://github.com/ubern-mia/bender/blob/main/training-models/dermamnist_v4_adam_TB.py) | SGD → Adam, TensorBoard logging | 0.762 | +0.01 |
| [v5](https://github.com/ubern-mia/bender/blob/main/training-models/dermamnist_v5_deeper_network.py) | Deeper 6-layer CNN (adds a 128-channel block) | 0.755 | −0.007 |
| [v6](https://github.com/ubern-mia/bender/blob/main/training-models/dermamnist_v6_even_deeper_network.py) | Even deeper 8-layer CNN (adds a 256-channel block) | < v5 | −0.01 |
| [v7](https://github.com/ubern-mia/bender/blob/main/training-models/dermamnist_v7_with_augm.py) | Data augmentation (flip + crop) on the 8-layer CNN | **0.770** | +0.015 |

Four things fall out of that table, and they generalise well beyond this dataset:

* **The biggest single win was a bug fix, not an idea.** v2 → v3 is worth ten percentage
  points, and all it did was set the learning rate to something sensible. Before you reach
  for a better architecture, check that the one you have is being optimised at all.
* **Adam bought robustness more than accuracy.** One point of accuracy, and much less
  sensitivity to getting the learning rate exactly right — which is why it is the safer
  default when you are still exploring.
* **More layers made it worse.** Twice. v5 and v6 both have lower training loss and lower
  test accuracy than v4: the extra capacity went into memorising the training set.
* **Regularization recovered it.** v7 is the deepest network *and* the best one, purely
  because augmentation stopped it memorising. Capacity is only useful when something
  prevents it being spent on the training set.

## Walking through the versions

### v1 — the naive baseline

A 4-layer CNN, SGD with a very small learning rate. It trains, the loss goes down, and the
result looks fine if you only read the headline number.

![Per-class precision and recall for version 1](../figures/training-models/dermamnist_v1_initial/per_class_metrics.png)

It is not fine. Only **melanocytic nevi** is being learned at all. That class is about
two-thirds of the dataset, so predicting it for everything yields a respectable-looking
weighted average while every other class scores zero. This is the
[episode 1](01-exploratory-data-analysis.md) warning about class imbalance, arriving on
schedule.

![Training loss, version 1](../figures/training-models/dermamnist_v1_initial/train_loss.png)

The training loss drops and then flattens. Once it flattens, further iterations cost time
and buy nothing.

![Validation accuracy, version 1](../figures/training-models/dermamnist_v1_initial/val_acc.png)

And the validation accuracy confirms it: stopping at 10,000 iterations would have given the
same result as running to 80,000. v3 introduces validation patience precisely to stop
paying for this.

### v2 — momentum 0.5 → 0.9

One line:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.000005, momentum=0.5)
# becomes
optimizer = torch.optim.SGD(model.parameters(), lr=0.000005, momentum=0.9)
```

![Training loss, version 2](../figures/training-models/dermamnist_v2_momentum0p9/train_loss.png)

The loss falls faster — it breaks 1.0 in under 10,000 iterations instead of 20,000.

![Validation accuracy, version 2](../figures/training-models/dermamnist_v2_momentum0p9/val_acc.png)

But the validation accuracy barely moves. Faster convergence to the same place. That is a
real result, and it says the bottleneck is somewhere else.

### v3 — the learning rate, and knowing when to stop

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
```

A thousandfold increase, plus **validation patience**: track validation accuracy against the
best seen so far, and if it fails to improve for a fixed number of evaluations, stop.

![Training loss, version 3](../figures/training-models/dermamnist_v3_lr0p005_val_patience/train_loss.png)

Noisier — larger steps — but much lower.

![Validation accuracy, version 3](../figures/training-models/dermamnist_v3_lr0p005_val_patience/val_acc.png)

And validation accuracy breaks 0.75. Ten points, from one hyperparameter that had been
wrong all along.

!!! tip "Find the learning rate first"
    Learning rate is the most sensitive hyperparameter in almost any deep learning setup,
    and the one most often left at whatever value was in the tutorial you copied. Sweep it
    across orders of magnitude — 1e-6, 1e-5, … 1e-1 — before touching anything else. It
    costs a few short runs and routinely beats weeks of architecture search.

### v4 — Adam, and logging that survives the project

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
```

Adam adapts the step size per parameter, which makes training much less sensitive to getting
the learning rate exactly right — [Karpathy's recipe](https://karpathy.github.io/2019/04/25/recipe/)
makes the case for it as the safe default. This version also introduces
[TensorBoard](https://pytorch.org/docs/stable/tensorboard.html), which is the change that
matters most for everything after it: from here on, comparing versions means overlaying
curves rather than squinting at printed numbers.

![Training and validation curves in TensorBoard, version 4](../figures/training-models/dermamnist_v4_adam_TB/train_val_TB_plots.png)

Read this one carefully, because it is the diagnostic picture of overfitting. Training loss
(bottom left) keeps falling and training accuracy (top left) keeps rising, while validation
loss (bottom right) plateaus and then *rises*, and validation accuracy (top right) plateaus
and slightly falls. The model is still learning — it is just learning the training set.

![Test accuracy and classification report, version 4](../figures/training-models/dermamnist_v4_adam_TB/test_accuracy_v4.png)

Test accuracy 0.762, and — more important than the number — every class now has non-zero
precision. Compare that to v1's single-class collapse.

### v5 and v6 — deeper, and worse

v5 adds a 128-channel block (6 layers); v6 adds a 256-channel block on top of that
(8 layers). Nothing else changes.

![v4 vs v5 training and validation curves](../figures/training-models/dermamnist_v5_deeper_network/training_val_v4_v5_comparison.png)

Blue is v4, grey is v5. Both losses are lower and training accuracy climbs faster — but
validation accuracy tracks v4 almost exactly. The extra capacity is going into the training
set.

![Test report, version 5](../figures/training-models/dermamnist_v5_deeper_network/test_accuracy_v5.png)

Test accuracy drops to 0.755. And look at `dermatofibroma`: all metrics zero. The model has
quietly given up on an entire class while posting a respectable average — which is why the
full classification report, not the headline number, is what you should be reading.

![v4 vs v5 vs v6](../figures/training-models/dermamnist_v6_even_deeper_network/training_val_comparison_v4_v5_v6.png)

Blue v4, pink v5, green v6. Same story, more so.

![Test report, version 6](../figures/training-models/dermamnist_v6_even_deeper_network/test_accuracy_v6.png)

Lower again. At this point the diagnosis is clear and the prescription follows from it: the
problem is not capacity, it is that nothing is stopping the network from memorising.

### v7 — augmentation, and beating the benchmark

```python
training_transform_medmnist = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Pad(2),
        transforms.RandomCrop(
            size=(32, 32), padding=(0, 0, 5, 5), padding_mode="reflect"
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ]
)
```

Every epoch now shows the network a slightly different version of each image, so memorising
individual training images stops being a winning strategy.

![v4 through v7 compared](../figures/training-models/dermamnist_v7_with_augm/training_val_v4_7.png)

Blue v4, cyan v5, pink v6, green v7. Note that v7 has the *lowest* training accuracy of the
four — and its validation accuracy does not fall away with more iterations. That gap closing
is what "overfitting solved" looks like on a chart.

![Test report, version 7](../figures/training-models/dermamnist_v7_with_augm/test_accuracy_v7.png)

**0.770** — above every DermaMNIST benchmark listed on the
[MedMNIST page](https://medmnist.com) at the time of writing. `dermatofibroma` is no longer
zero, and nearly every class has better precision than in any previous version.

!!! warning "Choosing augmentations is a domain decision, not a default"
    Horizontal and vertical flips are safe for dermatoscopic images: a skin lesion has no
    canonical orientation. They are *not* safe everywhere. Flip a chest X-ray left-to-right
    and you have created a patient with situs inversus. Rotate a brain MRI ninety degrees
    and you have created an image no scanner would ever produce. Ask what variation the
    real acquisition actually contains, and augment along those axes only.

## Looking inside the model

Each script has a visualization mode that skips training and writes three artifacts into
`results/vN/`:

```bash
make run-v4 MODE=visualize
```

| Artifact | Tool | What you get |
|---|---|---|
| `model_summary.txt` | [torchinfo](https://github.com/TylerYep/torchinfo) | Layer-by-layer table: output shapes, parameter counts, memory |
| `model_graph.png` | [torchview](https://github.com/mert-kurttutan/torchview) | A rendered dataflow graph (needs [Graphviz](https://graphviz.org)) |
| `model.onnx` | ONNX export | An interactive graph, via `make netron FILE=training-models/results/v4/model.onnx` |

[netron](https://netron.app) opens a browser tab where you can click through every layer and
inspect tensor shapes. It also works by dragging the `.onnx` file onto
[netron.app](https://netron.app).

## Try it yourself

```bash
make install
make run-v1     # the naive baseline
make run-v7     # the best version
```

| Artifact | What it is |
|---|---|
| [`dermamnist_v1_initial.ipynb`](https://github.com/ubern-mia/bender/blob/main/training-models/dermamnist_v1_initial.ipynb) · [Colab](https://colab.research.google.com/github/ubern-mia/bender/blob/main/training-models/dermamnist_v1_initial.ipynb) | The baseline as a notebook — the recommended starting point |
| [`dermamnist_v4_adam_TB.ipynb`](https://github.com/ubern-mia/bender/blob/main/training-models/dermamnist_v4_adam_TB.ipynb) · [Colab](https://colab.research.google.com/github/ubern-mia/bender/blob/main/training-models/dermamnist_v4_adam_TB.ipynb) | A second starting point, since v4's diff from v3 is larger than one line |
| [`dermamnist_v*.py`](https://github.com/ubern-mia/bender/tree/main/training-models) | All seven versions as scripts — `diff` two consecutive ones to see the change |
| [`shared/`](https://github.com/ubern-mia/bender/tree/main/training-models/shared) | The training loop, data loading and evaluation helpers the versions share |
| [`training-models/README.md`](https://github.com/ubern-mia/bender/blob/main/training-models/README.md) | The in-repository walkthrough this page is based on |

The most useful thing you can do is copy `dermamnist_v1_initial.ipynb` and apply each
version's change yourself, one at a time, watching the curves move. `diff v4 v5` is a
two-minute read and tells you exactly what "deeper network" cost.

!!! note "Where the outputs go"
    Checkpoints (`best_model.pt`), ONNX exports and TensorBoard event files are written to
    `results/vN/` and are **git-ignored**. The reference plots committed next to each script
    are what you see on this page. `make clean` removes `results/`.

## References and further reading

* **[A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)** —
  Andrej Karpathy, 2019. The general version of this episode; this one is its medical-imaging
  extension. Read it.
* **[MedMNIST v2](https://medmnist.com/)** — Yang et al., *Scientific Data*, 2023. The
  dataset and the benchmark table v7 beats.
* **[MONAI's 2D classification tutorial](https://github.com/Project-MONAI/tutorials/blob/main/2d_classification/mednist_tutorial.ipynb)** —
  the same kind of walkthrough using [MONAI](https://monai.io), which gives you medical-imaging-aware
  data loading and models out of the box.
* **[A beginner's guide to image segmentation with MRI and CT](https://arxiv.org/abs/2304.05901)** —
  from Leticia Rittner's group at UNICAMP, with code to follow along.
* **[Project roadmap for the medical imaging student working with deep learning](https://medium.com/miccai-educational-initiative/project-roadmap-for-the-medical-imaging-student-working-with-deep-learning-351add6066cf)** —
  another MICCAI Education Challenge entry, with more training-stage advice.
* **[MICCAI reproducibility checklist](https://github.com/JunMa11/MICCAI-Reproducibility-Checklist)** —
  run through it before you submit anything.
* **[itkwidgets](https://github.com/InsightSoftwareConsortium/itkwidgets)** and the
  [MONAI + itkwidgets guide](https://www.kitware.com/monai-and-itkwidgets-getting-started/) —
  medical image visualization inside notebooks.
* **[TensorBoardPlugin3D](https://www.kitware.com/tensorboardplugin3d-visualizing-3d-deep-learning-models-in-tensorboard/)** —
  volumetric visualization inside TensorBoard, from Kitware.
* **[MMAR](https://docs.nvidia.com/clara/clara-train-sdk/pt/mmar.html)** — NVIDIA's Medical
  Model Archive. Heavyweight for a prototype, useful once things stabilise.

---

**Next:** the benchmark is beaten. That turns out to be the easy part.
→ [Episode 4: Born to Deploy](04-evaluation-and-deployment.md)
