# Episode 5 — Becoming One with the Gradients

<div class="video-embed">
  <iframe src="https://www.youtube-nocookie.com/embed/hr1szGBP7Ps" title="BENDER Episode 5: Becoming one with the gradients" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

<div class="episode-meta">
  <span><strong>Runtime</strong> 15:12</span>
  <span><strong>Published</strong> 15 December 2022</span>
  <span><strong>Season</strong> Opens season 2, with new characters</span>
  <span><strong>Companion</strong> Notes and reading</span>
</div>

## What happens in the episode

A new season, and a new question: having built a model and evaluated it, can anyone say
*why* it decides what it decides? The episode goes after the black box in a deliberately
light-hearted way, and lands somewhere more careful than most introductions to the topic —
gradients are genuinely useful for understanding a model, and they are also genuinely
dangerous, because a gradient-based explanation can look completely convincing while telling
you nothing true.

## Why this matters

In medical imaging, interpretability is not a nice-to-have bolted on at the end. It shows up
in three separate places, each with different requirements:

* **Debugging.** Your model scores 0.95 AUC. Is it reading the pathology, or the
  laterality marker, or the fact that portable X-rays come from sicker patients? Saliency
  is often how you find out.
* **Clinical trust.** A radiologist asked to act on a prediction reasonably wants to know
  what it is looking at. "The network said so" is not a handover.
* **Regulation.** The EU AI Act classifies most clinical decision support as high-risk and
  requires transparency and human oversight; the FDA's guidance on machine-learning-enabled
  devices asks similar questions. This is now a compliance surface, not only a research one.

## A map of the terrain

The vocabulary in this field is a mess, largely because it grew in several communities at
once. A rough orientation:

| Distinction | What it means |
|---|---|
| **Interpretable** vs **explainable** | An *interpretable* model is understandable by construction (a short decision tree, a linear model on meaningful features). An *explainable* model is a black box with a separate method producing post-hoc explanations. These are not the same thing, and the difference matters — see Rudin below. |
| **Local** vs **global** | *Local*: why this prediction, for this image. *Global*: what has the model learned in general. Saliency maps are local; concept-based methods try to be global. |
| **Model-specific** vs **model-agnostic** | Grad-CAM needs the internals of a CNN. LIME and SHAP treat the model as a function you can query. |
| **Ante-hoc** vs **post-hoc** | Built into the model, or applied to a finished one. |

## Gradient-based methods, in ascending order of care

The family the episode's title refers to. All of them ask, in one form or another: *which
input pixels, if changed slightly, would most change the output?*

**Vanilla saliency** — take the gradient of the class score with respect to the input pixels
and visualise its magnitude. Simple, fast, and noisy: raw gradients on a deep ReLU network
are close to shattered.

**[Grad-CAM](https://arxiv.org/abs/1610.02391)** — weight the feature maps of the last
convolutional layer by the gradients flowing into them, then upsample. Far less noisy,
because it works at the resolution of semantic features rather than pixels. This is the
method you will see in most medical imaging papers, usually as a figure near the end.

**[Integrated Gradients](https://arxiv.org/abs/1703.01365)** — the gradient at one point is
arbitrary; integrate along a path from a baseline image (often black) to the actual image
instead. It satisfies attribution axioms that vanilla saliency does not. It also makes the
choice of baseline your problem, and for a CT scan "black" is not a neutral image — it is
air.

**Perturbation methods** — [LIME](https://arxiv.org/abs/1602.04938) and
[SHAP](https://arxiv.org/abs/1705.07874) occlude or perturb parts of the input and watch the
output move. Model-agnostic and slow, and SHAP comes with a solid game-theoretic grounding.

## The part most tutorials leave out

!!! danger "A saliency map that looks right is not evidence that the model is right"
    This is the episode's real message, and it is worth being blunt about.

    * **[Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292)** (Adebayo et
      al., NeurIPS 2018) randomised the model's weights and randomised the training labels.
      Several popular saliency methods produced *visually similar maps anyway*. A method
      whose output barely changes when the model is replaced with noise is not explaining
      that model.
    * **[Assessing the (Un)Trustworthiness of Saliency Maps for Localizing Abnormalities in Medical Imaging](https://arxiv.org/abs/2008.02766)**
      (Arun et al., *Radiology: AI*, 2021) ran that test on chest radiographs specifically.
      Most methods localised abnormalities worse than a trivial baseline, and were not
      repeatable across model instances.
    * **[The (Un)reliability of saliency methods](https://arxiv.org/abs/1711.00867)**
      (Kindermans et al.) shows attributions shifting under a constant offset to the input —
      a transformation the model's output is completely invariant to.

    None of this means "do not use saliency". It means: treat a saliency map as a
    *hypothesis to test*, not a result to publish. If it says the model is looking at the
    lesion, verify that by occluding the lesion and watching the prediction actually change.

And the deeper objection, from [Cynthia Rudin](https://arxiv.org/abs/1811.10154) (*Nature
Machine Intelligence*, 2019): for high-stakes decisions, stop explaining black boxes and use
interpretable models instead. A post-hoc explanation is by construction not a faithful
account of the computation — if it were, it would *be* the model. Whether that is practical
for imaging is a live argument, and it is the right argument to be having.

## Beyond saliency

* **Concept-based methods** — [TCAV](https://arxiv.org/abs/1711.11279) (Kim et al., ICML 2018) tests whether a
  human-nameable concept ("texture of a nevus", "presence of a marker") influences the
  prediction, and gives you a statistical answer rather than a picture. This tends to fit
  clinical questions better than pixel heatmaps.
* **Counterfactuals** — what minimal change to this image would flip the prediction? Often
  far more informative than "these pixels mattered", and increasingly generated with the
  models from [episode 7](07-generative-models.md).
* **Uncertainty** — a model that says "I do not know" is doing something adjacent to
  explaining itself, and it pairs naturally with the out-of-distribution concerns from
  [episode 4](04-evaluation-and-deployment.md).
* **Prototype and case-based models** — "this looks like these three training cases", which
  is closer to how a clinician actually reasons.

## Try it yourself

There is no companion script for this episode, but the models from
[episode 3](03-training-models.md) are a good target, and the exercise is short:

1. Train `dermamnist_v7_with_augm.py` and keep the checkpoint.
2. Attach [Captum](https://captum.ai) (PyTorch's interpretability library) and produce
   Grad-CAM and Integrated Gradients maps for a handful of test images.
3. Now run the sanity check: randomise the last layer's weights and regenerate the maps.
   **If they still look plausible, you have just reproduced Adebayo et al. on your own
   model** — and you will never look at a saliency figure the same way again.

## References and further reading

* **[A global taxonomy of interpretable AI: unifying the terminology for the technical and social sciences](https://link.springer.com/article/10.1007/s10462-022-10256-8)** —
  Graziani et al., *Artificial Intelligence Review*, 2023. The fastest way to stop being
  confused by the vocabulary, and the reference the episode points at.
* **[Introduction to Interpretable AI](https://www.i-aida.org/course/introinterpretableai/)** —
  an introductory course by Mara Graziani and colleagues, via AIDA.
* **[Interpretable ML video lectures](https://www.youtube.com/playlist?list=PLoROMvodv4rPh6wa6PGcHH6vMG9sEIPxL)** —
  Hima Lakkaraju, extending the first
  [Interpretable ML course at Harvard](https://interpretable-ml-class.github.io).
* **[iMIMIC workshop](https://imimic-workshop.com)** — Interpretability of Machine
  Intelligence in Medical Image Computing, at MICCAI. Talks and papers specifically in this
  domain.
* **[awesome-machine-learning-interpretability](https://github.com/jphall663/awesome-machine-learning-interpretability)** —
  Patrick Hall's compilation, thorough and kept current.
* **[Stop explaining black box machine learning models for high stakes decisions](https://arxiv.org/abs/1811.10154)** —
  Rudin, *Nature Machine Intelligence*, 2019.
* **[Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)** —
  Christoph Molnar's free book. The best single reference for the methods themselves.
* **[Captum](https://captum.ai)** — the library to actually implement any of this in PyTorch.

!!! info "BIAS — the Bern Interpretability AI Symposium"
    The episode was released alongside [BIAS 2023](https://amithjkamath.github.io/bias23/), a one-day
    symposium hosted at the University of Bern. Several talks from it are on the same
    YouTube channel, including keynotes by
    [Rich Caruana](https://www.youtube.com/watch?v=JJZk8YpjIu0),
    [Matt Lungren](https://www.youtube.com/watch?v=dkmgKe5kLPg) and
    [Henning Müller](https://www.youtube.com/watch?v=XgeeHPdOYSI), plus a
    [tutorial by Mara Graziani](https://www.youtube.com/watch?v=u3fY1wm0y7A).

---

**Next:** from classification to segmentation, and the architecture that dominated it for a
decade. → [Episode 6: The U-Net Model](06-u-net.md)
