# Reading list

Everything cited across the nine episodes, gathered in one place and grouped by what you
would be trying to do. Each episode page has its own, more contextual list — this is the
index.

## Start with these five

If you read nothing else from this site:

1. **[A visual introduction to machine learning](http://www.r2d3.us/)** — R2D3. Why looking at data first is not optional.
2. **[A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)** — Andrej Karpathy, 2019. The single best practical guide to not fooling yourself while training.
3. **[Machine learning for medical imaging: methodological failures and recommendations for the future](https://www.nature.com/articles/s41746-022-00592-y)** — Varoquaux & Cheplygina, *npj Digital Medicine*, 2022. How this goes wrong, catalogued.
4. **[nnU-Net](https://www.nature.com/articles/s41592-020-01008-z)** — Isensee et al., *Nature Methods*, 2021. The baseline you should beat before publishing an architecture.
5. **[Metrics reloaded](https://www.nature.com/articles/s41592-023-02151-z)** — Maier-Hein, Reinke et al., *Nature Methods*, 2024. How to choose a metric that answers your actual question.

## Data and preparation

* [Preparing Medical Imaging Data for Machine Learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC7104701/) — Willemink et al., *Radiology*, 2020.
* [DataPerf: Benchmarks for Data-Centric AI Development](https://arxiv.org/abs/2207.10062) — Mazumder et al., 2022.
* [Advances, challenges and opportunities in creating data for trustworthy AI](https://www.nature.com/articles/s42256-022-00516-1) — Liang et al., *Nature Machine Intelligence*, 2022.
* [The HAM10000 dataset](https://doi.org/10.1038/sdata.2018.161) — Tschandl et al., *Scientific Data*, 2018.
* [MedMNIST v2](https://medmnist.com/) — the dataset used throughout episodes 3 and 9.
* [BIDS](https://bids.neuroimaging.io) — Brain Imaging Data Structure.
* [Anatomical coordinate systems](https://www.slicer.org/wiki/Coordinate_systems#Anatomical_coordinate_system) — LPS vs RAS, explained.

## Training and experimental practice

* [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — Karpathy, 2019.
* [MONAI](https://monai.io) and its [tutorials](https://github.com/Project-MONAI/tutorials) — medical-imaging-aware PyTorch.
* [MICCAI reproducibility checklist](https://github.com/JunMa11/MICCAI-Reproducibility-Checklist).
* [Project roadmap for the medical imaging student working with deep learning](https://medium.com/miccai-educational-initiative/project-roadmap-for-the-medical-imaging-student-working-with-deep-learning-351add6066cf).
* [Nondeterminism and instability in neural network optimization](http://proceedings.mlr.press/v139/summers21a.html) — Summers & Dinneen, ICML 2021.
* [Unreproducible research is reproducible](http://proceedings.mlr.press/v97/bouthillier19a.html) — Bouthillier et al., ICML 2019.

## Evaluation, robustness and deployment

* [Metrics reloaded](https://www.nature.com/articles/s41592-023-02151-z) — Maier-Hein, Reinke et al., 2024, and its [preprint](https://arxiv.org/abs/2206.01653).
* [Common limitations of image processing metrics: a picture story](https://arxiv.org/abs/2104.05642) — Reinke, Tizabi et al.
* [Machine learning for medical imaging: methodological failures](https://www.nature.com/articles/s41746-022-00592-y) — Varoquaux & Cheplygina, 2022.
* [Improving out-of-distribution detection](https://ai.googleblog.com/2019/12/improving-out-of-distribution-detection.html) — Google AI blog.
* [scikit-learn's classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) — the minimum you should report.

## Interpretability

* [A global taxonomy of interpretable AI](https://link.springer.com/article/10.1007/s10462-022-10256-8) — Graziani et al., 2023.
* [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/) — Molnar's free book.
* [Grad-CAM](https://arxiv.org/abs/1610.02391) — Selvaraju et al., 2016.
* [Integrated Gradients](https://arxiv.org/abs/1703.01365) — Sundararajan et al., 2017.
* [Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292) — Adebayo et al., NeurIPS 2018.
* [Assessing the (Un)Trustworthiness of Saliency Maps in Medical Imaging](https://arxiv.org/abs/2008.02766) — Arun et al., *Radiology: AI*, 2021.
* [Stop explaining black box models for high stakes decisions](https://arxiv.org/abs/1811.10154) — Rudin, 2019.
* [Captum](https://captum.ai) — the PyTorch library.
* [iMIMIC workshop](https://imimic-workshop.com) and [awesome-machine-learning-interpretability](https://github.com/jphall663/awesome-machine-learning-interpretability).

## Segmentation

* [U-Net](https://arxiv.org/abs/1505.04597) — Ronneberger, Fischer & Brox, MICCAI 2015.
* [3D U-Net](https://arxiv.org/abs/1606.06650) — Çiçek et al., 2016, and [V-Net](https://arxiv.org/abs/1606.04797) — Milletari et al., 2016.
* [nnU-Net](https://www.nature.com/articles/s41592-020-01008-z) — Isensee et al., 2021, and its [code](https://github.com/MIC-DKFZ/nnUNet).
* [Loss odyssey in medical image segmentation](https://doi.org/10.1016/j.media.2021.102035) — Ma et al., 2021.
* [Medical image segmentation review: the success of U-Net](https://arxiv.org/abs/2211.14830) — Azad et al., 2022.
* [TransUNet](https://arxiv.org/abs/2102.04306), [UNETR](https://arxiv.org/abs/2103.10504), [Swin UNETR](https://arxiv.org/abs/2201.01266).
* [Medical Segmentation Decathlon](http://medicaldecathlon.com) — public tasks to benchmark on.

## Generative models

* [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Kingma & Welling, 2013.
* [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Goodfellow et al., 2014.
* [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Ho et al., 2020.
* [Latent diffusion](https://arxiv.org/abs/2112.10752) — Rombach et al., CVPR 2022.
* [GANs in medical imaging: a review](https://arxiv.org/abs/1809.07294) — Yi, Walia & Babyn, 2019.
* [Diffusion models in medical imaging: a comprehensive survey](https://doi.org/10.1016/j.media.2023.102846) — Kazerouni et al., 2023.
* [MONAI Generative Models](https://github.com/Project-MONAI/GenerativeModels).

## Foundation models

* [On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2108.07258) — Bommasani et al., 2021.
* [Foundation models for generalist medical AI](https://www.nature.com/articles/s41586-023-05881-4) — Moor et al., *Nature*, 2023.
* [Foundational Models in Medical Imaging: a survey](https://arxiv.org/abs/2310.18689) — Azad et al., 2023.
* [On the Challenges and Perspectives of Foundation Models for Medical Image Analysis](https://arxiv.org/abs/2306.05705) — Zhang & Metaxas, 2023.
* [MedSAM](https://arxiv.org/abs/2304.12306) — Ma et al., *Nature Communications*, 2024.
* [RETFound](https://www.nature.com/articles/s41586-023-06555-x) — Zhou et al., *Nature*, 2023.
* [Transfusion: Understanding Transfer Learning for Medical Imaging](https://arxiv.org/abs/1902.07208) — Raghu et al., NeurIPS 2019.

## Federated learning and privacy

* [FedAvg](https://arxiv.org/abs/1602.05629) — McMahan et al., AISTATS 2017.
* [The future of digital health with federated learning](https://www.nature.com/articles/s41746-020-00323-1) — Rieke et al., 2020.
* [Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977) — Kairouz et al., 2019.
* [FedProx](https://arxiv.org/abs/1812.06127) and [FedBN](https://arxiv.org/abs/2102.07623).
* [Deep Leakage from Gradients](https://arxiv.org/abs/1906.08935) and [Inverting Gradients](https://arxiv.org/abs/2003.14053).
* [EXAM](https://www.nature.com/articles/s41591-021-01506-3), [FeTS](https://www.nature.com/articles/s41467-022-33407-5), [Swarm Learning](https://www.nature.com/articles/s41586-021-03583-3).
* [Flower](https://flower.ai), [NVIDIA FLARE](https://github.com/NVIDIA/NVFlare), [OpenFL](https://github.com/securefederatedai/openfl), [Fed-BioMed](https://fedbiomed.org), [Opacus](https://opacus.ai).

## Writing it up

* [Microsoft Research: how to write a great research paper](https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/).
* [How we write rebuttals](https://deviparikh.medium.com/how-we-write-rebuttals-dc84742fece1) — Devi Parikh.
* [How not to be reviewer #2](https://link.springer.com/article/10.1007/s40037-021-00670-z).
