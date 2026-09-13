# Episode 6 — The U-Net Model

<div class="video-embed">
  <iframe src="https://www.youtube-nocookie.com/embed/AuDio_Clxo8" title="BENDER Episode 6: The U-Net Model" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

<div class="episode-meta">
  <span><strong>Runtime</strong> 18:28</span>
  <span><strong>Published</strong> 28 June 2023</span>
  <span><strong>Characters</strong> Lisa, Shelley, and a surprise guest</span>
  <span><strong>Companion</strong> Notes and an extensive reference list</span>
</div>

## What happens in the episode

Eight years after the paper, Lisa and Shelley take apart the architecture that ate medical
image segmentation. The episode covers the structure itself, why skip connections turn out
to be the load-bearing idea, the loss functions people actually use, the variants and
transformer descendants — and, characteristically for this series, a long detour into
reproducibility: the same U-Net, trained twice, does not give you the same model.

## Why this matters

Classification asks *what is in this image*. Segmentation asks *which voxels* — and in
clinical work that is usually the question. Tumour volume, organ-at-risk contours for
radiotherapy planning, lesion burden over time: all of them need a per-voxel answer, not a
label.

That makes segmentation a dense prediction problem, and dense prediction has two
requirements that pull against each other:

* **Semantic context.** To know that a region is tumour, the network needs a wide receptive
  field — it has to see the surroundings.
* **Spatial precision.** To place the boundary correctly, it needs high resolution.

Downsampling buys context and destroys resolution. The U-Net's answer is to do both, and
then stitch them back together.

## The architecture

[U-Net](https://arxiv.org/abs/1505.04597) (Ronneberger, Fischer & Brox, MICCAI 2015) is an
encoder–decoder with a symmetric shape — hence the U — and one crucial addition.

```
input ──► conv ──► conv ─────────────────────────────────────────► conv ──► conv ──► output
            │                   skip connection (concatenate)         ▲
            ▼                                                         │
          pool ──► conv ──► conv ───────────────────────► upconv ─────┘
                     │          skip connection              ▲
                     ▼                                       │
                   pool ──► conv ──► conv ──► upconv ────────┘
                              (bottleneck)
```

* The **contracting path** (encoder) is a normal CNN: convolutions and pooling, halving
  resolution and doubling channels at each level. It builds context.
* The **expanding path** (decoder) mirrors it, upsampling back to full resolution.
* The **skip connections** concatenate each encoder feature map onto the decoder feature map
  at the same resolution. This is the part that matters.

### Why skip connections are the idea

Without them, everything the decoder knows has to squeeze through the bottleneck, and fine
spatial detail does not survive the trip — you get roughly correct blobs with wrong
boundaries. The skip connections hand the decoder the high-resolution features directly, so
it can combine *where things are* (from early layers) with *what they are* (from deep
layers).

They also make the network far easier to optimise, by giving gradients a short path back to
the early layers.
[Drozdzal et al.](https://link.springer.com/chapter/10.1007/978-3-319-46976-8_19) examined
this specifically for biomedical segmentation and found both effects. Interestingly, the
same property has a downside: [Wu et al.](https://arxiv.org/abs/2002.05990) showed that skip
connections make adversarial examples *more* transferable between networks — a good
reminder that no architectural choice is purely free.

### Why it worked so well for medical images

* **It trains on very little data.** The original paper used ~30 annotated images, leaning
  hard on augmentation (including elastic deformations, which are far more plausible for
  tissue than for natural images).
* **It is fully convolutional**, so it handles varying input sizes and tiles naturally.
* **It extends to 3D** almost unchanged — swap in 3D convolutions and you have
  [3D U-Net](https://arxiv.org/abs/1606.06650) or
  [V-Net](https://arxiv.org/abs/1606.04797).
* **It is small enough to train on one GPU**, which in 2015 mattered a great deal and in a
  hospital research group still does.

## Loss functions: the other half of the problem

Segmentation losses deserve as much thought as the architecture, because
[episode 4's](04-evaluation-and-deployment.md) class-imbalance problem is far worse per-voxel
than per-image. A lesion may be 0.1% of a volume; plain cross-entropy will happily declare
everything background and score 99.9%.

| Loss | Idea | Watch out for |
|---|---|---|
| Cross-entropy | Per-voxel classification | Dominated by the background class |
| Weighted / focal CE | Reweight rare classes or hard voxels | Another hyperparameter to tune |
| Dice loss | Directly optimises overlap | Unstable on empty or tiny targets |
| Dice + CE | The pragmatic default, used by nnU-Net | — |
| Boundary / Hausdorff losses | Penalise boundary error specifically | Usually need a stable loss to warm up first |

[Ma et al., "Loss odyssey in medical image segmentation"](https://doi.org/10.1016/j.media.2021.102035)
(*Medical Image Analysis*, 2021) compares twenty of them systematically and is the reference
to reach for. Its practical conclusion is refreshingly boring: compound Dice + cross-entropy
is hard to beat.

## The reproducibility detour

A substantial part of the episode is about something that sounds like a footnote and is not:
**train the same U-Net twice and you get two different models.** Nondeterminism in GPU
kernels, data ordering, and initialisation all contribute, and the resulting spread in test
metrics is often the same size as the difference between the methods people publish.

* [Summers & Dinneen](http://proceedings.mlr.press/v139/summers21a.html), ICML 2021, on
  nondeterminism and instability in neural network optimization.
* [Bouthillier et al.](http://proceedings.mlr.press/v97/bouthillier19a.html), ICML 2019 —
  "Unreproducible research is reproducible": you can reproduce the *run* and still not
  reproduce the *finding*.
* [Alahmari et al.](https://doi.org/10.1109/ACCESS.2020.3039833), IEEE Access 2020, on the
  repeatability of deep learning models specifically.
* [Mosbach et al.](https://arxiv.org/abs/2006.04884) and
  [Du et al.](https://arxiv.org/abs/2302.07778) on how unstable fine-tuning is.

The practical consequence, and it is the same message as [episode 4](04-evaluation-and-deployment.md):
**report mean and standard deviation over several seeds.** A single-seed improvement of half
a Dice point is not a result.

## What came after

**[nnU-Net](https://www.nature.com/articles/s41592-020-01008-z)** (Isensee et al., *Nature
Methods*, 2021) is the most important entry here, and the most under-appreciated. Rather
than proposing a better architecture, it proposes a *procedure*: automatically configure
preprocessing, patch size, batch size, augmentation and postprocessing from the properties
of the dataset, around a nearly vanilla U-Net. It then proceeded to win or match
state-of-the-art on dozens of segmentation challenges.

!!! tip "Start with nnU-Net"
    If you have a segmentation task and a deadline, run [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
    first. It is the strongest baseline available, it takes very little effort, and if your
    clever architecture cannot beat it you have learned that early and cheaply. A great many
    published architectures cannot.

**Transformer-based variants.** After [ViT](https://arxiv.org/abs/2010.11929) showed that
[attention](https://arxiv.org/abs/1706.03762) could replace convolution for images, the
U-shape absorbed it:

* [TransUNet](https://arxiv.org/abs/2102.04306) — a transformer encoder feeding a U-Net decoder.
* [UNETR](https://arxiv.org/abs/2103.10504) — a pure transformer encoder for 3D volumes.
* [Swin UNETR](https://arxiv.org/abs/2201.01266) — hierarchical shifted-window attention, strong on brain tumour segmentation.

Whether they actually beat a well-tuned CNN is genuinely contested — see
[Gut et al.](https://doi.org/10.1109/TMI.2022.3180435) for a careful benchmark, and note
that nnU-Net remains competitive with all of them. Transformers also bring their own
robustness questions under distribution shift
([Zhang et al.](https://arxiv.org/abs/2106.07617),
[Bai et al.](https://arxiv.org/abs/2111.05464)).

**Foundation models for segmentation.** [SAM](https://arxiv.org/abs/2304.02643) and
[MedSAM](https://arxiv.org/abs/2304.12306) point at a different future entirely, which is
[episode 8](08-foundation-models.md)'s subject.

## Try it yourself

There is no companion script for this episode. The most useful next steps, roughly in order
of effort:

1. **Read the 8-page [original paper](https://arxiv.org/abs/1505.04597).** It is unusually
   clear, and short.
2. **Implement one from scratch.** A 2D U-Net is about 60 lines of PyTorch, and writing the
   skip connections yourself is worth more than reading ten explanations of them.
3. **Use [MONAI's `UNet`](https://docs.monai.io/en/stable/networks.html)** for anything real
   — it handles 2D/3D, residual units and the medical-imaging data pipeline around it.
4. **Run [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)** on a public dataset from the
   [Medical Segmentation Decathlon](http://medicaldecathlon.com) to see what a strong
   baseline actually looks like.
5. **Then evaluate it properly** — with [Metrics Reloaded](https://www.nature.com/articles/s41592-023-02151-z),
   per-structure, over multiple seeds.

## References and further reading

The video description carries a long bibliography; these are the entries worth starting
with, grouped.

**The architecture**

* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — Ronneberger, Fischer & Brox, MICCAI 2015.
* [The importance of skip connections in biomedical image segmentation](https://link.springer.com/chapter/10.1007/978-3-319-46976-8_19) — Drozdzal et al., 2016.
* [Skip connections matter: on the transferability of adversarial examples generated with ResNets](https://arxiv.org/abs/2002.05990) — Wu et al., 2020.
* [LinkNet: exploiting encoder representations for efficient semantic segmentation](https://arxiv.org/abs/1707.03718) — Chaurasia & Culurciello, 2017.

**Reviews**

* [Medical image segmentation review: the success of U-Net](https://arxiv.org/abs/2211.14830) — Azad et al., 2022. The comprehensive one.
* [U-Net and its variants for medical image segmentation: a review of theory and applications](https://doi.org/10.1109/ACCESS.2021.3086020) — Siddique et al., *IEEE Access*, 2021.

**Losses and configuration**

* [Loss odyssey in medical image segmentation](https://doi.org/10.1016/j.media.2021.102035) — Ma et al., *Medical Image Analysis*, 2021.
* [nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation](https://www.nature.com/articles/s41592-020-01008-z) — Isensee et al., *Nature Methods*, 2021.
* [Benchmarking of deep architectures for segmentation of medical images](https://doi.org/10.1109/TMI.2022.3180435) — Gut et al., *IEEE TMI*, 2022.

**Reproducibility**

* [Nondeterminism and instability in neural network optimization](http://proceedings.mlr.press/v139/summers21a.html) — Summers & Dinneen, ICML 2021.
* [Unreproducible research is reproducible](http://proceedings.mlr.press/v97/bouthillier19a.html) — Bouthillier et al., ICML 2019.
* [Challenges for the repeatability of deep learning models](https://doi.org/10.1109/ACCESS.2020.3039833) — Alahmari et al., *IEEE Access*, 2020.
* [On the stability of fine-tuning BERT](https://arxiv.org/abs/2006.04884) — Mosbach et al., 2020.
* [Measuring the instability of fine-tuning](https://arxiv.org/abs/2302.07778) — Du et al., 2023.
* [Machine learning for medical imaging: methodological failures and recommendations for the future](https://www.nature.com/articles/s41746-022-00592-y) — Varoquaux & Cheplygina, 2022.

**Evaluation**

* [Metrics reloaded: recommendations for image analysis validation](https://www.nature.com/articles/s41592-023-02151-z) — Maier-Hein, Reinke et al., *Nature Methods*, 2024.
* [Common limitations of image processing metrics: a picture story](https://arxiv.org/abs/2104.05642) — Reinke, Tizabi et al.

**Transformers and what came next**

* [Attention is all you need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017.
* [An image is worth 16×16 words](https://arxiv.org/abs/2010.11929) — Dosovitskiy et al., 2020.
* [TransUNet](https://arxiv.org/abs/2102.04306) — Chen et al., 2021.
* [UNETR: transformers for 3D medical image segmentation](https://arxiv.org/abs/2103.10504) — Hatamizadeh et al., WACV 2022.
* [Swin UNETR](https://arxiv.org/abs/2201.01266) — Hatamizadeh et al., 2022.
* [Transforming medical imaging with transformers?](https://doi.org/10.1016/j.media.2023.102762) — Li et al., *Medical Image Analysis*, 2023.
* [Delving deep into the generalization of vision transformers under distribution shifts](https://arxiv.org/abs/2106.07617) — Zhang et al., CVPR 2022.
* [Are transformers more robust than CNNs?](https://arxiv.org/abs/2111.05464) — Bai et al., NeurIPS 2021.
* [High-resolution image synthesis with latent diffusion models](https://arxiv.org/abs/2112.10752) — Rombach et al., CVPR 2022 — which leads directly into [episode 7](07-generative-models.md).

---

**Next:** enough discriminating. What if the model produced images instead?
→ [Episode 7: Generative Models in Medical Imaging](07-generative-models.md)
