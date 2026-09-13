# Episode 4 — Born to Deploy

<div class="video-embed">
  <iframe src="https://www.youtube-nocookie.com/embed/YwM7qwqSy9k" title="BENDER Episode 4: Born to Deploy" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

<div class="episode-meta">
  <span><strong>Runtime</strong> 5:38</span>
  <span><strong>Published</strong> 25 August 2022</span>
  <span><strong>Characters</strong> Satish, Mike and colleagues</span>
  <span><strong>Companion</strong> Notes and reading</span>
</div>

## What happens in the episode

Satish has a model that beats the published benchmark. He is ready to show the clinicians
and start writing the paper. Then the model meets the ultimate test — data it has not seen,
from somewhere it has not been — and the episode is about everything between "it works on my
test set" and "it is worth putting in front of a patient".

This is the last episode of the original four-part MICCAI Education Challenge entry, and it
is deliberately the one that takes the win away.

## Why this matters

A held-out test split answers one narrow question: *how does this model perform on data
drawn from the same distribution as its training data?* In medical imaging that question is
almost never the one you care about, because deployment is by definition a different
distribution — a different scanner, a different protocol, a different population, a
different year.

Everything below is a way of asking a harder and more useful question.

## External test data

Portability across clinical settings is the thing. In deployment your model will see
out-of-distribution data, subjects of different geography and ethnicity, and a dozen other
variations that were simply absent from your training set. A state-of-the-art model can fail
badly on all of them.

Wherever separately held-out data from such a setting exists — another hospital, another
cohort, another year — evaluate on it *before* you claim anything. An external test set that
degrades your headline number is not a bad result. It is the first honest one.

## Multiple vendors and varied protocols

Training on data from one scanner, in one hospital, under one acquisition protocol, is
**single-source bias**. The risk is not abstract: the model may learn the nuances of that
specific machine and protocol more readily than the actual characteristics of the pathology
you wanted it to learn. Reconstruction kernels, field strength, slice thickness and vendor
defaults are all strong, consistent signals — and a network will happily use any signal that
predicts the label in your training set.

So vary those parameters deliberately when you evaluate: different hardware vendors, a
diverse set of acquisition protocols (sometimes simulable), and a test set chosen for
*breadth of coverage of the deployment distribution*, not only for label accuracy.

## Average metrics are not robustness

Datasets are biased, including curated public ones. Deep networks are extremely good at
learning patterns — not necessarily the patterns a human would associate with the
categories you care about. Achieving the highest average metric on the test set is not
sufficient evidence of reliability; it may only mean the model has incrementally learned to
perform better *on that test set*, which in a challenge or leaderboard setting has a name:
overfitting the test set.

More informative things to report:

* **Standard deviation across multiple training runs**, not one seed's best number.
* **Performance under deliberate worst cases** — artificially degraded inputs, plausible
  artefacts, the awkward subgroup.
* **Adversarial or augmentation-based robustness checks**, where feasible.

This is a lot of work, and it is not always possible. It is also the difference between a
number and a claim.

## Imbalanced categories, and reporting them honestly

As [episode 3](03-training-models.md) demonstrated at length, medical image data is prone to
class imbalance, and segmentation is worse: the pixels belonging to a lesion are a tiny
minority of the pixels in the volume. There are modelling remedies, but the *reporting*
obligation is separate and simpler:

!!! warning "Report per-class metrics, not just the average"
    Give the full confusion matrix, or at minimum a
    [classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html).
    Version 5 in [episode 3](03-training-models.md) scored a respectable 0.755 while giving
    `dermatofibroma` a precision, recall and F1 of exactly zero. The average never showed
    it. The per-class table showed it immediately — and that is the kind of failure that
    surfaces in deployment, in front of someone who is counting on it.

For segmentation, the same principle applies to metric choice itself: Dice on a small
structure behaves very differently from Dice on a large one, and a single averaged Dice
across a dataset can hide catastrophic failure on the rare cases that matter most. The
[Metrics Reloaded](https://www.nature.com/articles/s41592-023-02151-z) framework is the
current best guide to picking metrics that match the question.

## A practical pre-submission checklist

- [ ] Evaluated on at least one **external** dataset, and reported the drop.
- [ ] Reported **per-class** or per-structure metrics, not only averages.
- [ ] Reported **variance across seeds**, not a single best run.
- [ ] Checked performance across **subgroups** — scanner, site, sex, age, skin tone where relevant.
- [ ] Confirmed the metric actually matches the clinical question (see Metrics Reloaded).
- [ ] Documented the **failure modes** you found, rather than only the successes.
- [ ] Fixed the random seed, pinned dependencies, and made the run reproducible.
- [ ] Stated clearly what the model is **not** validated for.

## Dealing with reviewer #2

And then you have to write it up, where [reviewer #2](https://twitter.com/GrumpyReviewer2)
is waiting. Some help from the community:

* [Microsoft Research: how to write a great research paper](https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/) — Simon Peyton Jones's talk and slides.
* [Devi Parikh on writing rebuttals](https://deviparikh.medium.com/how-we-write-rebuttals-dc84742fece1)
* [How not to be reviewer #2 yourself](https://link.springer.com/article/10.1007/s40037-021-00670-z)

## References and further reading

* **[Machine learning for medical imaging: methodological failures and recommendations for the future](https://www.nature.com/articles/s41746-022-00592-y)** —
  Varoquaux & Cheplygina, *npj Digital Medicine*, 2022. The best single summary of how these
  failures happen and what to do instead.
* **[Improving out-of-distribution detection](https://ai.googleblog.com/2019/12/improving-out-of-distribution-detection.html)** —
  Google AI blog, on knowing when your model is being shown something it should refuse to
  answer.
* **[Preparing Medical Imaging Data for Machine Learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC7104701/)** —
  Willemink et al., *Radiology*, 2020. A thorough tour of acquisition, vendor and protocol
  variation, and what each of them does to a model.
* **[Metrics Reloaded: recommendations for image analysis validation](https://www.nature.com/articles/s41592-023-02151-z)** —
  Maier-Hein, Reinke et al., *Nature Methods* 21:195–212, 2024
  ([preprint](https://arxiv.org/abs/2206.01653)). If you are choosing a segmentation or
  detection metric, start here — there is an online tool that walks you through it.
* **[Common limitations of image processing metrics: a picture story](https://arxiv.org/abs/2104.05642)** —
  Reinke, Tizabi et al. The illustrated companion — every page is a way your metric can mislead you.
* **[MICCAI reproducibility checklist](https://github.com/JunMa11/MICCAI-Reproducibility-Checklist)** —
  work through it before submitting.

---

**Next:** the model is evaluated. But *why* does it decide what it decides?
→ [Episode 5: Becoming One with the Gradients](05-interpretability.md)
