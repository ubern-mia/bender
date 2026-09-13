# Episode 1 — The Data Pile

<div class="video-embed">
  <iframe src="https://www.youtube-nocookie.com/embed/NtszpkE0gc4" title="BENDER Episode 1: The Data Pile" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

<div class="episode-meta">
  <span><strong>Runtime</strong> 8:06</span>
  <span><strong>Published</strong> 3 August 2022</span>
  <span><strong>Characters</strong> Satish, Mike</span>
  <span><strong>Companion</strong> <a href="../../resources/checklist/">Data checklist</a> · two notebooks</span>
</div>

## What happens in the episode

Satish has finally been given the clinical data for his project, and he is delighted. The
episode is about the half-hour after that delight wears off: the folder is not one dataset,
it is a pile. Slices in inconsistent orientations. Spacings that differ per subject.
Metadata nobody mentioned. A few files that should never have left the hospital with the
patient's name still attached.

Mike's advice is the whole point of the series in miniature: **look at your data before you
model it**. Not "run `describe()` on it" — actually open it, plot it, and check that it is
what someone told you it was.

## Why this matters

In natural-image computer vision you can mostly trust that a JPEG is a JPEG. Medical images
carry a great deal more structure, and every piece of that structure is a place where an
assumption can go wrong:

* **Physical geometry.** A voxel is not a pixel. It has a size in millimetres, and that
  size varies between subjects, scanners and protocols. A network trained on arrays alone
  is being shown lesions at inconsistent physical scale without being told.
* **Orientation and coordinate systems.** LPS, RAS, and the several conventions that differ
  only in sign. Get one axis backwards and you have silently trained a model on mirrored
  anatomy — which, for anything lateralized, is a real clinical error and an entirely
  invisible bug.
* **Metadata that leaks.** DICOM tags carry acquisition parameters (useful), and also names,
  birth dates and accession numbers (very much not yours to have).
* **Labels produced by humans under time pressure.** Segmentation masks drawn by three
  different residents to three different implicit protocols are three different datasets.

None of this is exotic. All of it is routine, and all of it is cheaper to find now than
after three weeks of training runs.

### The thing that makes this different from a Kaggle dataset

A public benchmark has been cleaned by someone whose job was to clean it. Data straight
from a clinical archive has been *stored*, not *curated* — it was organised for clinical
retrieval, not for machine learning. Those are different goals, and the gap between them is
your first week of work.

!!! warning "Class imbalance is the default, not the exception"
    In DermaMNIST, melanocytic nevi make up roughly two-thirds of every split. A model that
    predicts "nevus" for every single image scores about 67% accuracy while having learned
    nothing at all. You will watch exactly this happen in
    [episode 3](03-training-models.md), and it comes back again in
    [episode 9](09-federated-learning.md). Check your class histogram on day one.

## The checklist

The companion artifact for this episode is a one-page checklist of things to verify *after*
you have access to clinical imaging data and *before* you build anything with it. In short:

1. **List dimensions and spacings** for every subject, and save them to a CSV.
2. **Build a reproducible conversion pipeline** from raw to processed — keep the two
   physically separate.
3. **Reorient and resample to a standard space** — and never use linear interpolation on
   label masks, because it invents label values that do not exist.
4. **Confirm anonymization.** If you find identifiable information, tell your supervisor
   rather than quietly deleting it.
5. **Plot the data.** Check the categories are right, the masks land on the right anatomy,
   and there are no duplicates or motion artefacts.
6. **Check for multi-source bias** if the data came from more than one place.
7. **Consider atlas registration** as further spatial normalization, where it applies.

[Read the full checklist :material-arrow-right:](../resources/checklist.md){ .md-button }

A [downloadable PDF version](https://github.com/ubern-mia/bender/blob/main/exploratory-data-analysis/checklist.pdf)
is in the repository if you would rather print it and pin it somewhere.

## Try it yourself

| Artifact | What it shows |
|---|---|
| [`explore_dermamnist.ipynb`](https://github.com/ubern-mia/bender/blob/main/exploratory-data-analysis/explore_dermamnist.ipynb) · [open in Colab](https://colab.research.google.com/github/ubern-mia/bender/blob/main/exploratory-data-analysis/explore_dermamnist.ipynb) | Loading [MedMNIST](https://medmnist.com/), plotting samples, and — the important part — counting the per-class distribution before assuming anything about it |
| [`explore_dicom.ipynb`](https://github.com/ubern-mia/bender/blob/main/exploratory-data-analysis/explore_dicom.ipynb) · [open in Colab](https://colab.research.google.com/github/ubern-mia/bender/blob/main/exploratory-data-analysis/explore_dicom.ipynb) | Reading DICOM tags with [pydicom](https://pydicom.github.io/): spacing, orientation, scanner model, and what is hiding in the header |
| [`checklist.md`](https://github.com/ubern-mia/bender/blob/main/exploratory-data-analysis/checklist.md) | The checklist above, in the repository |

If you are working with your own data rather than MedMNIST, run the DICOM notebook against
a handful of your own series first. The tags that turn out to be missing, or inconsistent,
tell you more about the dataset than any summary statistic will.

## Tools worth knowing

| Tool | For |
|---|---|
| [3D Slicer](https://www.slicer.org) / [ITK-SNAP](http://www.itksnap.org) | Actually looking at volumes and masks, interactively |
| [SimpleITK](https://simpleitk.readthedocs.io/) / [pydicom](https://pydicom.github.io/) | Reading, resampling and reorienting in Python |
| [deid](https://github.com/pydicom/deid) | Anonymizing DICOM headers systematically |
| [cookiecutter data science](https://drivendata.github.io/cookiecutter-data-science/#directory-structure) | A project layout that keeps raw, interim and processed data apart |
| [BIDS](https://bids.neuroimaging.io) | A real standard for organising neuroimaging studies — worth conforming to if it fits |
| [Duplicate Image Finder](https://github.com/elisemercury/Duplicate-Image-Finder) | Catching the same subject twice under two names |

## References and further reading

* **[A visual introduction to machine learning](http://www.r2d3.us/)** — R2D3. Not medical
  imaging at all, and still the single best argument for looking at your data before
  modelling it. Start here if you read only one thing.
* **[DataPerf: Benchmarks for Data-Centric AI Development](https://arxiv.org/abs/2207.10062)** —
  Mazumder et al., 2022.
* **[Advances, challenges and opportunities in creating data for trustworthy AI](https://www.nature.com/articles/s42256-022-00516-1)** —
  Liang et al., *Nature Machine Intelligence*, 2022.
* **[Andrew Ng, AI Minimalist: The Machine-Learning Pioneer Says Small is the New Big](https://ieeexplore.ieee.org/document/9754503)** —
  *IEEE Spectrum*, 2022, on the data-centric turn.
* **[The HAM10000 dataset](https://doi.org/10.1038/sdata.2018.161)** — Tschandl et al.,
  *Scientific Data*, 2018. The source of the DermaMNIST images used throughout this series.
* **[Anatomical coordinate systems](https://www.slicer.org/wiki/Coordinate_systems#Anatomical_coordinate_system)** —
  the 3D Slicer wiki page that explains LPS vs RAS properly.

---

**Next:** you have the data, and you have questions about it. Now you have to ask the
clinicians. → [Episode 2: Meet the Experts](02-terminology.md)
