---
hide:
  - navigation
---

# The :robot: BENDER series

**BE**st practices in medical imagi**N**g **DE**ep lea**R**ning — nine short videos, and
the code, checklists and notes that go with them.

Are you a new(ish) graduate or a very enthusiastic undergraduate, working with medical
image data, and wondering how to get that deep learning model to train on it? This is for
you. Follow Satish, Mike and friends through the ups and downs of building a
state-of-the-art model, and learn not to make the same mistakes they do.

[Watch the playlist :fontawesome-brands-youtube:](https://www.youtube.com/playlist?list=PLFwdflE4leRpqIz-F68pvwFATIOEwrSHp){ .md-button .md-button--primary }
[Start here :material-rocket-launch:](getting-started.md){ .md-button }

The series was submitted to the [MICCAI Education Challenge 2022](https://miccai-sb.github.io/challenge.html)
and has kept going since. Each episode below has its own page: what happens in the video,
an introduction to the topic itself, what you can run from this repository, and where to
read further.

## The episodes

<div class="episode-grid">

<a class="episode-card" href="episodes/01-exploratory-data-analysis/">
  <img src="https://i.ytimg.com/vi/NtszpkE0gc4/hqdefault.jpg" alt="Episode 1 thumbnail" loading="lazy">
  <span class="episode-card__body">
    <span class="episode-card__eyebrow">Episode 1 · 8 min</span>
    <span class="episode-card__title">The Data Pile</span>
    <span class="episode-card__desc">Satish gets his clinical data and discovers that "the data" is never one thing. A checklist for everything to look at before you model.</span>
  </span>
</a>

<a class="episode-card" href="episodes/02-terminology/">
  <img src="https://i.ytimg.com/vi/jGLBcMyiehg/hqdefault.jpg" alt="Episode 2 thumbnail" loading="lazy">
  <span class="episode-card__body">
    <span class="episode-card__eyebrow">Episode 2 · 5 min</span>
    <span class="episode-card__title">Meet the Experts</span>
    <span class="episode-card__desc">The meeting with the clinicians goes sideways. A two-way glossary for talking to people who do not share your jargon.</span>
  </span>
</a>

<a class="episode-card" href="episodes/03-training-models/">
  <img src="https://i.ytimg.com/vi/f0wd8EvRiH0/hqdefault.jpg" alt="Episode 3 thumbnail" loading="lazy">
  <span class="episode-card__body">
    <span class="episode-card__eyebrow">Episode 3 · 8 min</span>
    <span class="episode-card__title">Good Model Training Shall You Strive For</span>
    <span class="episode-card__desc">Seven versions of one model, changing one thing at a time, from 0.65 to 0.77 test accuracy. The longest episode in code.</span>
  </span>
</a>

<a class="episode-card" href="episodes/04-evaluation-and-deployment/">
  <img src="https://i.ytimg.com/vi/YwM7qwqSy9k/hqdefault.jpg" alt="Episode 4 thumbnail" loading="lazy">
  <span class="episode-card__body">
    <span class="episode-card__eyebrow">Episode 4 · 6 min</span>
    <span class="episode-card__title">Born to Deploy</span>
    <span class="episode-card__desc">The model works in the notebook. Now: other scanners, other hospitals, other patients — and reviewer #2.</span>
  </span>
</a>

<a class="episode-card" href="episodes/05-interpretability/">
  <img src="https://i.ytimg.com/vi/hr1szGBP7Ps/hqdefault.jpg" alt="Episode 5 thumbnail" loading="lazy">
  <span class="episode-card__body">
    <span class="episode-card__eyebrow">Episode 5 · 15 min</span>
    <span class="episode-card__title">Becoming One with the Gradients</span>
    <span class="episode-card__desc">Opening the black box. Saliency maps, what they actually show, and the ways they quietly mislead.</span>
  </span>
</a>

<a class="episode-card" href="episodes/06-u-net/">
  <img src="https://i.ytimg.com/vi/AuDio_Clxo8/hqdefault.jpg" alt="Episode 6 thumbnail" loading="lazy">
  <span class="episode-card__body">
    <span class="episode-card__eyebrow">Episode 6 · 18 min</span>
    <span class="episode-card__title">The U-Net Model</span>
    <span class="episode-card__desc">The architecture that ate medical image segmentation — how it works, why skip connections matter, and what came after.</span>
  </span>
</a>

<a class="episode-card" href="episodes/07-generative-models/">
  <img src="https://i.ytimg.com/vi/Bp3OUSdtkfY/hqdefault.jpg" alt="Episode 7 thumbnail" loading="lazy">
  <span class="episode-card__body">
    <span class="episode-card__eyebrow">Episode 7 · 20 min</span>
    <span class="episode-card__title">Generative Models in Medical Imaging</span>
    <span class="episode-card__desc">A Matrix-themed tour of VAEs, GANs, conditional GANs and diffusion models — and when synthetic data helps.</span>
  </span>
</a>

<a class="episode-card" href="episodes/08-foundation-models/">
  <img src="https://i.ytimg.com/vi/JVfEAjbw5hk/hqdefault.jpg" alt="Episode 8 thumbnail" loading="lazy">
  <span class="episode-card__body">
    <span class="episode-card__eyebrow">Episode 8 · 13 min</span>
    <span class="episode-card__title">Foundation Models for Medical Imaging</span>
    <span class="episode-card__desc">Pre-train once, adapt everywhere. What foundation models promise in healthcare, and what they still cost.</span>
  </span>
</a>

<a class="episode-card" href="episodes/09-federated-learning/">
  <img src="https://i.ytimg.com/vi/i2kaDRe8BDo/hqdefault.jpg" alt="Episode 9 thumbnail" loading="lazy">
  <span class="episode-card__body">
    <span class="episode-card__eyebrow">Episode 9 · 8 min</span>
    <span class="episode-card__title">Federated Learning in Medical Imaging</span>
    <span class="episode-card__desc">Two hospitals, two continents, no data transfer. Simulate a five-hospital federation on your laptop.</span>
  </span>
</a>

</div>

## What you can actually run

The series is not only videos. Everything below is in the repository and runs on a laptop
CPU — see [Start here](getting-started.md) for setup.

| Artifact | Episode | What it is |
|---|---|---|
| [`explore_dermamnist.ipynb`](https://github.com/ubern-mia/bender/blob/main/exploratory-data-analysis/explore_dermamnist.ipynb) | [1](episodes/01-exploratory-data-analysis.md) | Looking at a dataset properly before modelling it |
| [`explore_dicom.ipynb`](https://github.com/ubern-mia/bender/blob/main/exploratory-data-analysis/explore_dicom.ipynb) | [1](episodes/01-exploratory-data-analysis.md) | Reading the non-image metadata in DICOM files |
| [Data checklist](resources/checklist.md) | [1](episodes/01-exploratory-data-analysis.md) | Seven things to verify before you train anything |
| [Glossary](resources/glossary.md) | [2](episodes/02-terminology.md) | Clinical terms for engineers, technical terms for clinicians |
| [`dermamnist_v1` … `v7`](https://github.com/ubern-mia/bender/tree/main/training-models) | [3](episodes/03-training-models.md) | Seven training scripts, one change at a time |
| [`dermamnist_federated.py`](https://github.com/ubern-mia/bender/blob/main/federated-learning/dermamnist_federated.py) | [9](episodes/09-federated-learning.md) | A simulated five-hospital federation over the same data |

## How to use this

!!! tip "Watch, then run, then break something"
    Each episode page is written so you can read it on its own. But the series is meant to
    be *followed*: watch the episode, read the page, then copy a notebook and change one
    thing in it. Episode 3 in particular is built around changing exactly one line at a
    time and watching what happens — which is the single habit that separates a debuggable
    experiment from a mystery.

If you spot something wrong, or something missing, please
[open an issue](https://github.com/ubern-mia/bender/issues). We would much rather hear
about it.

:wave: The [Medical Image Analysis group](https://www.artorg.unibe.ch/research/mia/) at
Universität Bern
