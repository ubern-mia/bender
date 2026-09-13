# Generative Models in Medical Imaging

> 📖 **This page has a fuller companion on the website:** [Episode 7 — Generative Models in Medical Imaging](https://ubern-mia.github.io/bender/episodes/07-generative-models/) — with the video, an introduction to the topic, and a grouped reading list.

Welcome to Episode 07 of the [BENDER Series](https://github.com/ubern-mia/bender). Having spent episodes 1 through 6 learning to *discriminate* — is this lesion malignant, which pixels are tumour — we now turn the problem around and ask a model to *generate*: to produce images that could plausibly have come from the scanner, but never did.

Inspired by the iconic Matrix movie, which blends deep philosophical questions with AI technology (a highly recommended watch), we've tried to capture some of that spirit in this episode. We go through the basic concepts of generative models and then dive into Variational Auto-Encoders, Generative Adversarial Networks, Conditional GANs, their pros and cons, their use in medical imaging applications, and the basics of Diffusion Models.

--------------------

## The basic concept

A discriminative model learns `p(y | x)`: given this image, which class? A generative model learns `p(x)` itself — the distribution the images were drawn from — so that you can sample new ones from it. That shift sounds academic until you notice what it buys you in medical imaging: augmentation for classes you barely have, anomaly detection by asking how badly a "normal" model reconstructs a case, cross-modality translation, and shareable synthetic cohorts when the real ones cannot leave the hospital (which is exactly where [episode 9](../federated-learning/README.md) picks up).

## Variational Auto-Encoders (VAEs)

An autoencoder squeezes an image through a bottleneck and reconstructs it. A [Variational Auto-Encoder](https://arxiv.org/abs/1312.6114) (Kingma & Welling) makes that bottleneck *probabilistic*: the encoder outputs a distribution rather than a point, and a KL term pulls that distribution towards a standard normal. The payoff is a latent space you can actually sample from and interpolate through, rather than a lookup table with gaps.

**Pros:** stable to train, principled probabilistic formulation, a latent space that means something — useful on its own for clustering and anomaly scoring.

**Cons:** samples are famously blurry. The pixel-wise reconstruction loss rewards hedging, and a hedged prediction of a fine texture is a smooth one. For imaging tasks where the diagnostic signal *is* the fine texture, that blur is not a cosmetic problem.

## Generative Adversarial Networks (GANs)

[GANs](https://arxiv.org/abs/1406.2661) (Goodfellow et al.) replace the hand-written loss with a second network. A generator produces images from noise; a discriminator tries to tell real from fake; each improves by defeating the other. Nobody ever writes down what "realistic" means — the discriminator learns it.

The result is sharp samples, and a training process that can fall over in interesting ways: mode collapse (the generator finds one convincing output and stops exploring), vanishing gradients, and the general absence of a loss value that tells you whether things are going well.

* **[DC-GAN](https://arxiv.org/abs/1511.06434)** — the architectural recipe that first made GAN training reliably reproducible: all-convolutional, batch norm, no fully connected layers.
* **[WGAN](https://arxiv.org/abs/1701.07875)** — swaps the Jensen-Shannon objective for the Wasserstein distance, giving gradients that stay informative and a loss that actually correlates with sample quality.
* **[Conditional GAN](https://arxiv.org/abs/1411.1784)** — feeds a label (or another image) to both networks, so you can ask for *a melanoma* rather than *something*. This is the variant most medical imaging work needs: image-to-image translation, modality synthesis, and class-conditional augmentation are all conditional problems.
* **The [GAN Zoo](https://github.com/hindupuravinash/the-gan-zoo)** — for a sense of just how many `<X>GAN` variants have been proposed. It is a long list.

## GANs for medical imaging

The community has reviewed this space thoroughly, and the surveys are a better starting point than any single paper:

* [GAN-based generation of realistic 3D volumetric data: a systematic review and taxonomy](https://doi.org/10.1016/j.media.2024.103100) — Ferreira et al., Medical Image Analysis, 2024.
* [Data synthesis and adversarial networks: a review and meta-analysis in cancer imaging](https://doi.org/10.1016/j.media.2022.102704) — Osuala et al., Medical Image Analysis, 2022.
* [Generative adversarial networks in medical image augmentation: a review](https://doi.org/10.1016/j.compbiomed.2022.105382) — Chen et al., Computers in Biology and Medicine, 2022.
* [Generative adversarial networks in medical image segmentation: a review](https://doi.org/10.1016/j.compbiomed.2021.105063) — Xun et al., Computers in Biology and Medicine, 2022.
* [Systematic review of GANs for medical image classification and segmentation](https://link.springer.com/article/10.1007/s10278-021-00556-w) — Jeong et al., Journal of Digital Imaging, 2022.
* [Generative adversarial network in medical imaging: a review](https://arxiv.org/abs/1809.07294) — Yi, Walia & Babyn, Medical Image Analysis, 2019: still the best single entry point.
* [A survey on deep learning applied to medical images: from simple artificial neural networks to generative models](https://link.springer.com/article/10.1007/s00521-022-07953-4) — Celard et al., Neural Computing and Applications, 2022.
* [Generative adversarial networks in computer vision: a survey and taxonomy](https://arxiv.org/abs/1906.01529) — Wang, She & Ward, 2019.

## Diffusion Models

The current state of the art takes a different route again: [destroy an image by adding Gaussian noise over many small steps, then train a network to undo one step of that](https://arxiv.org/abs/2006.11239) (Ho et al., DDPM). Generation is running the learned denoiser backwards from pure noise. Training is stable (it is just denoising regression — no adversarial game), coverage of the data distribution is better than a GAN's, and sample quality is excellent. The price is sampling speed: hundreds of forward passes per image, though this is improving quickly.

* [Diffusion models in medical imaging: a comprehensive survey](https://doi.org/10.1016/j.media.2023.102846) — Kazerouni et al., Medical Image Analysis, 2023.
* [Diffusion models in vision: a survey](https://arxiv.org/abs/2209.04747) — Croitoru et al., IEEE TPAMI, 2023.
* [MONAI Generative Models](https://github.com/Project-MONAI/GenerativeModels) — implementations of VAEs, GANs and diffusion models built for medical imaging, now folded into core MONAI.

## A word of caution before you synthesize your training set

Synthetic data is seductive, and it is worth being clear-eyed about what it can and cannot do:

* A generative model trained on your data cannot add information that was not in it. Augmenting a rare class with samples from a model that itself saw only 12 examples of that class mostly amplifies those 12.
* Synthetic images inherit the biases of the training set — including the single-source scanner bias we flagged in [episode 4](../evaluating-and-deploying-model/README.md).
* Generative models can memorize. "Synthetic" is not automatically "anonymous", and a patient-identifiable sample is a patient-identifiable sample regardless of how it was produced.
* Evaluate on **real** held-out data, always. FID and friends measure whether images look plausible, not whether they are clinically faithful — a reader study or a downstream-task metric is what actually answers that.

--------------------

Feel free to share your experiences and challenges with these technologies for medical image computing — for questions/suggestions for improvements, please [create an issue](https://github.com/ubern-mia/bender/issues) in the BENDER repository.
