# Episode 2 — Meet the Experts

<div class="video-embed">
  <iframe src="https://www.youtube-nocookie.com/embed/jGLBcMyiehg" title="BENDER Episode 2: Meet the Experts" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

<div class="episode-meta">
  <span><strong>Runtime</strong> 5:23</span>
  <span><strong>Published</strong> 11 August 2022</span>
  <span><strong>Characters</strong> Satish, Mike, the clinical experts</span>
  <span><strong>Companion</strong> <a href="../../resources/glossary/">Two-way glossary</a></span>
</div>

## What happens in the episode

Satish and Mike go to meet their clinical collaborators, armed with questions from
[episode 1](01-exploratory-data-analysis.md). The meeting does not go the way anyone
planned. Both sides are competent, both sides are speaking English, and neither side
understands the other — because "contrast", "resolution", "significant" and "validation"
all mean something specific and *different* on either side of the table.

It is the shortest episode in the series and, judging by the response to it, the one people
recognise themselves in most.

## Why this matters

Most failures in a clinical-AI collaboration are not modelling failures. They are
translation failures that only surface months later:

* You ask for "the tumour segmentation". You get the *gross tumour volume*. Your
  radiation-oncology collaborator assumed you meant the *planning target volume*, which is
  a different contour drawn for a different purpose with a margin around it.
* You report that your model is "significant". They hear *clinically* significant. You meant
  p < 0.05 on a held-out split of 200 patients.
* They say the scan is "T1". You train on it. Half the series turn out to be T1 *with
  contrast*, which is a visibly different image and, for enhancing lesions, an entirely
  different problem.
* You say "validation set". They hear *clinical validation* — the thing that involves an
  ethics board and a prospective study.

None of these are anybody's fault. They are what happens when two fields share vocabulary
but not definitions, and the fix is not cleverness — it is writing the definitions down.

!!! tip "Three questions that save months"
    1. **"Can you show me on a scan what you mean?"** Ambiguity in words survives; ambiguity
       in a pointed finger does not.
    2. **"Who drew these labels, to what protocol, and would a second reader draw them the
       same way?"** Inter-rater variability is the noise floor on your metric. If two
       experts agree on a Dice of 0.85, your model scoring 0.87 is not beating anyone.
    3. **"What decision would this model change?"** If nothing in the clinical workflow
       moves, the project is an exercise. Better to know at the start.

## Speaking both languages

The companion artifact is a glossary that runs in both directions — because the
misunderstanding is symmetric.

**Clinical terms for technical folks:** anatomical directions (superior, inferior, medial,
lateral…), planes of the body (axial, coronal, sagittal), what CT and MRI actually measure,
functional MRI, MRI sequences and what T1 / T2 / FLAIR / T1c mean, contrast agents,
hyper- and hypo-intense, and the difference between a lesion, a tumour and a target volume.

**Technical terms for clinical folks:** AI, machine learning and deep learning (which are
not synonyms), neural networks, CNNs, classification vs segmentation, U-Net, DenseNet,
GANs, Siamese networks, transfer learning, federated learning, skip connections,
super-resolution, data augmentation, regularization, optimizers, and validation patience.

[Read the full glossary :material-arrow-right:](../resources/glossary.md){ .md-button }

A [downloadable PDF version](https://github.com/ubern-mia/bender/blob/main/terminology-meet-experts/glossar.pdf)
is in the repository — several groups have told us they hand it to new students on day one,
which is exactly what it is for.

## Practical habits for the collaboration

* **Keep a shared definitions document.** One page, plain language, both sides edit it. Add
  a row every time a term causes confusion. It will be the most-referenced file in the
  project.
* **Annotate together once.** Sit with the clinician for one hour while they segment three
  cases and narrate what they are doing. You will learn more about your labels than from any
  paper describing them.
* **Show them model outputs early and badly.** A clinician looking at your week-three
  failure cases will spot the systematic error in seconds. A clinician looking at your
  month-six ROC curve will say "very nice".
* **Ask what they would not trust.** Their scepticism tells you which evaluation to build
  — which is exactly the territory of [episode 4](04-evaluation-and-deployment.md).

## References and further reading

* **[The glossary](../resources/glossary.md)** — the companion page, and its
  [PDF](https://github.com/ubern-mia/bender/blob/main/terminology-meet-experts/glossar.pdf).
* **[SEER training: anatomical terminology](https://training.seer.cancer.gov/anatomy/body/terminology.html)** —
  the NCI's free primer on directions and planes. Twenty minutes well spent.
* **[NIBIB science education glossary](https://www.nibib.nih.gov/science-education/glossary)** —
  plain-language definitions of the imaging modalities, from the NIH.
* **[A panel discussion on exactly this problem](https://youtu.be/Gbnep6RJinQ?t=1626)** —
  timestamped to the relevant part.
* **[Machine learning for medical imaging: methodological failures and recommendations for the future](https://www.nature.com/articles/s41746-022-00592-y)** —
  Varoquaux & Cheplygina, *npj Digital Medicine*, 2022. Several of the failures it catalogues
  begin as vocabulary problems.

---

**Next:** definitions agreed, data understood. Time to actually train something — and to do
it in a way you can explain later. → [Episode 3: Good Model Training Shall You Strive For](03-training-models.md)
