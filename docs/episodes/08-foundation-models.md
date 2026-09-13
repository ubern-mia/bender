# Episode 8 — Foundation Models for Medical Imaging

<div class="video-embed">
  <iframe src="https://www.youtube-nocookie.com/embed/JVfEAjbw5hk" title="BENDER: Foundation Models for Medical Imaging" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

<div class="episode-meta">
  <span><strong>Runtime</strong> 12:54</span>
  <span><strong>Published</strong> 21 November 2024</span>
  <span><strong>Topic</strong> Pre-training, adaptation, and what it costs</span>
  <span><strong>Companion</strong> Notes and reading</span>
</div>

## What happens in the episode

Every episode so far has trained a model for one task on one dataset. This one asks what
happens when you invert that: train one very large model on an enormous amount of unlabelled
data, then adapt it to your task with a fraction of the labels you would otherwise need.

The episode covers what foundation models are, the pre-training strategies that make them
work, how you adapt one, and — at some length — what makes healthcare a harder place to
deploy them than most.

## What is a foundation model?

The term was coined by [Bommasani et al. (2021)](https://arxiv.org/abs/2108.07258) for a
model **trained on broad data at scale, usually self-supervised, that can be adapted to a
wide range of downstream tasks**.

Three properties matter:

1. **Scale** — orders of magnitude more data than any single task's labelled set.
2. **Self-supervision** — the training signal comes from the data itself, so labels stop
   being the bottleneck. For medical imaging, where a labelled volume can cost an expert an
   hour, this is the entire point.
3. **Adaptability** — one pre-trained model serves many tasks, via fine-tuning, a linear
   probe, or just a prompt.

This is different in degree, not in kind, from the ImageNet pre-training that medical
imaging has relied on for a decade — but the degree is large enough to change what is
possible. And ImageNet transfer to medical images was always a slightly awkward fit;
[Raghu et al.](https://arxiv.org/abs/1902.07208) showed it often buys less than assumed.

## How they are pre-trained

**Contrastive learning** — pull augmented views of the same image together in feature space
and push different images apart ([SimCLR](https://arxiv.org/abs/2002.05709),
[MoCo](https://arxiv.org/abs/1911.05722)). Works well; requires care in choosing
augmentations, since the "different image" may be the same patient's other slice.

**Masked image modelling** — hide patches and reconstruct them
([MAE](https://arxiv.org/abs/2111.06377)). Scales well with vision transformers and needs no
negative pairs. This is what [RETFound](https://www.nature.com/articles/s41586-023-06555-x)
used on 1.6 million retinal images.

**Vision–language pre-training** — this is medical imaging's unfair advantage. Every clinical
image already has a free text label attached to it: the radiology report. Align the two, as
[CLIP](https://arxiv.org/abs/2103.00020) does for web images, and you get a model that
understands the vocabulary of the domain without anyone annotating anything.
[MedCLIP](https://arxiv.org/abs/2210.10163) and
[BiomedCLIP](https://arxiv.org/abs/2303.00915) are the medical instances;
[CheXzero](https://www.nature.com/articles/s41551-022-00936-9) showed this can reach
radiologist-level chest X-ray classification with *no* explicit labels at all.

**Promptable segmentation** — [SAM](https://arxiv.org/abs/2304.02643) was trained on a
billion masks to segment whatever a user points at.
[MedSAM](https://arxiv.org/abs/2304.12306) (Ma et al., *Nature Communications*, 2024)
re-trained that idea on over 1.5 million medical image–mask pairs across modalities.

## How you adapt one

| Method | What it does | When |
|---|---|---|
| **Linear probe** | Freeze the encoder, train a classifier on its features | Very little labelled data; also the honest test of the representation |
| **Full fine-tuning** | Update everything | Plenty of data and compute; risks catastrophic forgetting |
| **Parameter-efficient fine-tuning** ([LoRA](https://arxiv.org/abs/2106.09685), adapters) | Train a small number of extra parameters | The usual sweet spot — most of the benefit, a fraction of the cost, and you can keep several task-specific adapters over one base model |
| **Prompting / zero-shot** | No training at all | SAM-style interactive use, or vision-language zero-shot classification |

!!! tip "Try the linear probe first"
    It takes minutes, needs almost no labelled data, and tells you whether the foundation
    model's features are actually useful for *your* task before you spend a week on
    fine-tuning. If a linear probe on frozen features cannot beat your
    [episode 3](03-training-models.md) baseline, a bigger model is not your problem.

## Where they have worked

* **[RETFound](https://www.nature.com/articles/s41586-023-06555-x)** (Zhou et al., *Nature*,
  2023) — retinal images, self-supervised on 1.6M images, then adapted to predict ocular
  *and systemic* disease with far fewer labels.
* **[MedSAM](https://arxiv.org/abs/2304.12306)** — universal promptable segmentation across
  modalities.
* **Computational pathology** — [UNI](https://www.nature.com/articles/s41591-024-02857-3)
  and [CONCH](https://www.nature.com/articles/s41591-024-02856-4) (Chen, Lu et al., *Nature
  Medicine*, 2024), trained on very large slide collections. Pathology has benefited
  unusually well, because whole-slide images provide an enormous amount of unlabelled data.
* **[CheXzero](https://www.nature.com/articles/s41551-022-00936-9)** — label-free chest
  radiograph classification from reports alone.

## The honest part

This is the section the episode spends the most time on, and rightly.

* **"Foundation model" is not a performance claim.** A well-configured
  [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) still beats a fine-tuned foundation model on
  many segmentation tasks. Compare against a strong task-specific baseline, not against a
  weak one.
* **Zero-shot rarely means zero-shot in the clinic.** MedSAM segments what you point at; it
  does not know which structure you wanted, and interactive prompting is not free in a
  radiologist's workflow.
* **Bias scales too.** A model pre-trained on data from a handful of institutions carries
  their population, their scanners, and their documentation habits into every downstream
  task built on it — and now it does so invisibly, because you did not assemble that
  dataset and probably cannot inspect it.
* **Evaluation is genuinely unsolved.** If the pre-training corpus is undisclosed, you cannot
  rule out that your test set was in it. In medical imaging, where the same public datasets
  circulate widely, this is a real and common problem rather than a theoretical one.
* **Provenance and consent.** Data that was legitimately collected for one study is not
  automatically available for pre-training a general-purpose model. Governance here is still
  being worked out.
* **Cost and access.** Pre-training is out of reach for almost every hospital research group.
  That concentrates capability in a few places — which is part of why
  [episode 9](09-federated-learning.md)'s question about where data lives matters more, not
  less.

!!! warning "The reflex to develop"
    When someone reports that their foundation model achieves an excellent score, the first
    question is not *how large was it* but **what was it pre-trained on, and can you show me
    that my test set was not in there?** Very often the answer is no, and the number should
    be read accordingly.

## Try it yourself

No companion script, but three concrete things you can do today:

1. **Linear-probe a public foundation model on DermaMNIST.** Take BiomedCLIP or any
   self-supervised encoder, freeze it, extract features for the
   [episode 3](03-training-models.md) data, and fit a logistic regression. Compare against
   v7's 0.770. The comparison is the exercise.
2. **Run [MedSAM](https://github.com/bowang-lab/MedSAM) on your own images.** Its
   repository has a straightforward demo. Watch where it works and, more usefully, where it
   does not.
3. **Try zero-shot classification with a vision-language model** and see how far text
   prompts alone get you before any training.

## References and further reading

**Framing**

* [On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2108.07258) — Bommasani et al., 2021. Where the term comes from.
* [Foundation models for generalist medical artificial intelligence](https://www.nature.com/articles/s41586-023-05881-4) — Moor et al., *Nature*, 2023. The vision, stated carefully.
* [Foundational Models in Medical Imaging: A Comprehensive Survey and Future Vision](https://arxiv.org/abs/2310.18689) — Azad et al., 2023.
* [On the Challenges and Perspectives of Foundation Models for Medical Image Analysis](https://arxiv.org/abs/2306.05705) — Zhang & Metaxas, 2023. The sceptical companion piece.

**Pre-training methods**

* [A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)](https://arxiv.org/abs/2002.05709) — Chen et al., 2020.
* [Momentum Contrast (MoCo)](https://arxiv.org/abs/1911.05722) — He et al., 2019.
* [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) — He et al., 2021.
* [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — Radford et al., 2021.
* [Transfusion: Understanding Transfer Learning for Medical Imaging](https://arxiv.org/abs/1902.07208) — Raghu et al., NeurIPS 2019. On how little ImageNet pre-training actually buys.

**Medical instances**

* [MedCLIP: Contrastive Learning from Unpaired Medical Images and Text](https://arxiv.org/abs/2210.10163) — Wang et al., EMNLP 2022.
* [BiomedCLIP](https://arxiv.org/abs/2303.00915) — Zhang et al., 2023.
* [Segment Anything](https://arxiv.org/abs/2304.02643) — Kirillov et al., ICCV 2023.
* [Segment Anything in Medical Images (MedSAM)](https://arxiv.org/abs/2304.12306) — Ma et al., *Nature Communications*, 2024.
* [A foundation model for generalizable disease detection from retinal images (RETFound)](https://www.nature.com/articles/s41586-023-06555-x) — Zhou et al., *Nature*, 2023.
* [Expert-level detection of pathologies from unannotated chest X-ray images (CheXzero)](https://www.nature.com/articles/s41551-022-00936-9) — Tiu et al., *Nature Biomedical Engineering*, 2022.
* [Towards a general-purpose foundation model for computational pathology (UNI)](https://www.nature.com/articles/s41591-024-02857-3) — Chen et al., *Nature Medicine*, 2024.

**Adaptation**

* [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021. The technique, not only for language.

---

**Next:** the final episode, and the one with code you can run — training across hospitals
that are not allowed to share data.
→ [Episode 9: Federated Learning in Medical Imaging](09-federated-learning.md)
