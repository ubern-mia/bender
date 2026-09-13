# Episode 7 — Generative Models in Medical Imaging

<div class="video-embed">
  <iframe src="https://www.youtube-nocookie.com/embed/Bp3OUSdtkfY" title="BENDER: Generative Models in Medical Imaging" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

<div class="episode-meta">
  <span><strong>Runtime</strong> 20:15</span>
  <span><strong>Published</strong> 2 April 2024</span>
  <span><strong>Theme</strong> The Matrix</span>
  <span><strong>Companion</strong> Notes and reading</span>
</div>

Inspired by the Matrix — which blends deep philosophical questions with AI technology, and
is a highly recommended watch — this is the longest and most thoroughly chaptered episode in
the series.

## Chapters

| | |
|---|---|
| [00:00](https://www.youtube.com/watch?v=Bp3OUSdtkfY&t=0s) | Introduction |
| [03:04](https://www.youtube.com/watch?v=Bp3OUSdtkfY&t=184s) | The basic concept of generative models |
| [04:32](https://www.youtube.com/watch?v=Bp3OUSdtkfY&t=272s) | Variational Auto-Encoders (VAEs) |
| [08:10](https://www.youtube.com/watch?v=Bp3OUSdtkfY&t=490s) | Pros and cons of VAEs |
| [08:40](https://www.youtube.com/watch?v=Bp3OUSdtkfY&t=520s) | Generative Adversarial Networks (GANs) |
| [11:58](https://www.youtube.com/watch?v=Bp3OUSdtkfY&t=718s) | DC-GANs |
| [13:50](https://www.youtube.com/watch?v=Bp3OUSdtkfY&t=830s) | WGANs |
| [15:12](https://www.youtube.com/watch?v=Bp3OUSdtkfY&t=912s) | Conditional GANs |
| [16:24](https://www.youtube.com/watch?v=Bp3OUSdtkfY&t=984s) | The GAN Zoo |
| [16:45](https://www.youtube.com/watch?v=Bp3OUSdtkfY&t=1005s) | GANs for medical imaging |
| [17:48](https://www.youtube.com/watch?v=Bp3OUSdtkfY&t=1068s) | Diffusion models |
| [18:56](https://www.youtube.com/watch?v=Bp3OUSdtkfY&t=1136s) | Outro |

## The basic concept

Having spent episodes 1 through 6 learning to *discriminate* — is this lesion malignant,
which voxels are tumour — this episode turns the problem around and asks a model to
*generate*: to produce images that could plausibly have come from the scanner, but never
did.

A discriminative model learns `p(y | x)`: given this image, which class? A generative model
learns `p(x)` itself — the distribution the images were drawn from — so that you can sample
new ones from it.

That shift sounds academic until you notice what it buys you in medical imaging:

* **Augmentation** for classes you barely have.
* **Anomaly detection**, by asking how badly a model of "normal" reconstructs a case.
* **Cross-modality translation** — synthesising CT from MR for radiotherapy planning, say.
* **Shareable synthetic cohorts**, when the real ones cannot leave the hospital — which is
  exactly where [episode 9](09-federated-learning.md) picks up.

## Variational Auto-Encoders

An autoencoder squeezes an image through a bottleneck and reconstructs it. A
[Variational Auto-Encoder](https://arxiv.org/abs/1312.6114) (Kingma & Welling, 2013) makes
that bottleneck *probabilistic*: the encoder outputs a distribution rather than a point, and
a KL term pulls that distribution towards a standard normal. The payoff is a latent space
you can actually sample from and interpolate through, rather than a lookup table with gaps.

**Pros:** stable to train, principled probabilistic formulation, and a latent space that
means something — useful on its own for clustering and anomaly scoring.

**Cons:** samples are famously blurry. The pixel-wise reconstruction loss rewards hedging,
and a hedged prediction of a fine texture is a smooth one. For imaging tasks where the
diagnostic signal *is* the fine texture, that blur is not a cosmetic problem.

## Generative Adversarial Networks

[GANs](https://arxiv.org/abs/1406.2661) (Goodfellow et al., 2014) replace the hand-written
loss with a second network. A generator produces images from noise; a discriminator tries to
tell real from fake; each improves by defeating the other. Nobody ever writes down what
"realistic" means — the discriminator learns it.

The result is sharp samples, and a training process that can fall over in interesting ways:
mode collapse (the generator finds one convincing output and stops exploring), vanishing
gradients, and the general absence of a loss value that tells you whether things are going
well.

* **[DC-GAN](https://arxiv.org/abs/1511.06434)** — the architectural recipe that first made
  GAN training reliably reproducible: all-convolutional, batch norm, no fully connected
  layers.
* **[WGAN](https://arxiv.org/abs/1701.07875)** — swaps the Jensen–Shannon objective for the
  Wasserstein distance, giving gradients that stay informative and a loss that actually
  correlates with sample quality.
* **[Conditional GAN](https://arxiv.org/abs/1411.1784)** — feeds a label (or another image)
  to both networks, so you can ask for *a melanoma* rather than *something*. This is the
  variant most medical imaging work needs: image-to-image translation, modality synthesis
  and class-conditional augmentation are all conditional problems. See also
  [pix2pix](https://arxiv.org/abs/1611.07004) and
  [CycleGAN](https://arxiv.org/abs/1703.10593) for the unpaired case, which comes up
  constantly when the two modalities were never acquired on the same patient.
* **The [GAN Zoo](https://github.com/hindupuravinash/the-gan-zoo)** — for a sense of just
  how many `<X>GAN` variants have been proposed. It is a long list.

## Diffusion models

The current state of the art takes a different route again:
[destroy an image by adding Gaussian noise over many small steps, then train a network to
undo one step of that](https://arxiv.org/abs/2006.11239) (Ho et al., DDPM, 2020). Generation
is running the learned denoiser backwards from pure noise.

Training is stable — it is just denoising regression, with no adversarial game — coverage of
the data distribution is better than a GAN's, and sample quality is excellent. The price is
sampling speed: hundreds of forward passes per image, though this is improving quickly.
[Latent diffusion](https://arxiv.org/abs/2112.10752) (Rombach et al., 2022) moves the
diffusion process into a compressed latent space, which is what made the approach practical
at high resolution — and what most medical imaging work now builds on.

Note the architecture at the heart of almost every diffusion model: a **U-Net**, from
[episode 6](06-u-net.md). It never left.

## A word of caution before you synthesize your training set

Synthetic data is seductive, and it is worth being clear-eyed about what it can and cannot
do.

!!! danger "Four things to hold on to"
    * **A generative model cannot add information that was not in its training data.**
      Augmenting a rare class with samples from a model that itself saw only 12 examples of
      that class mostly amplifies those 12.
    * **Synthetic images inherit the biases of the training set** — including the
      single-source scanner bias flagged in [episode 4](04-evaluation-and-deployment.md).
    * **Generative models can memorize.** "Synthetic" is not automatically "anonymous", and
      a patient-identifiable sample is patient-identifiable regardless of how it was
      produced.
    * **Evaluate on real held-out data, always.** FID and friends measure whether images
      look plausible, not whether they are clinically faithful. A reader study or a
      downstream-task metric is what actually answers that.

## Try it yourself

No companion script for this episode, but the ground is well covered:

* **[MONAI Generative Models](https://github.com/Project-MONAI/GenerativeModels)** —
  implementations of VAEs, GANs and diffusion models built for medical imaging, now folded
  into [core MONAI](https://monai.io). The tutorials there are the fastest honest start.
* **A VAE on DermaMNIST** is a genuinely good weekend exercise: reuse the data loading from
  [episode 3](03-training-models.md)'s [`shared/data.py`](https://github.com/ubern-mia/bender/blob/main/training-models/shared/data.py),
  and see how blurry 28×28 skin lesions get.
* **Then run the real test:** train the [episode 3](03-training-models.md) classifier on
  synthetic data alone and evaluate it on the *real* test set. That number tells you what
  your generator actually learned, and it is usually humbling.

## References and further reading

**Foundations**

* [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Kingma & Welling, 2013.
* [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Goodfellow et al., 2014.
* [Unsupervised representation learning with deep convolutional GANs](https://arxiv.org/abs/1511.06434) — Radford, Metz & Chintala, 2015.
* [Wasserstein GAN](https://arxiv.org/abs/1701.07875) — Arjovsky, Chintala & Bottou, 2017.
* [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — Mirza & Osindero, 2014.
* [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Ho, Jain & Abbeel, 2020.
* [High-resolution image synthesis with latent diffusion models](https://arxiv.org/abs/2112.10752) — Rombach et al., CVPR 2022.

**Surveys for medical imaging** — the community has reviewed this space thoroughly, and the
surveys are a better starting point than any single paper.

* [Generative adversarial network in medical imaging: a review](https://arxiv.org/abs/1809.07294) — Yi, Walia & Babyn, *Medical Image Analysis*, 2019. Still the best single entry point.
* [GAN-based generation of realistic 3D volumetric data: a systematic review and taxonomy](https://doi.org/10.1016/j.media.2024.103100) — Ferreira et al., *Medical Image Analysis*, 2024.
* [Data synthesis and adversarial networks: a review and meta-analysis in cancer imaging](https://doi.org/10.1016/j.media.2022.102704) — Osuala et al., *Medical Image Analysis*, 2023.
* [Generative adversarial networks in medical image augmentation: a review](https://doi.org/10.1016/j.compbiomed.2022.105382) — Chen et al., *Computers in Biology and Medicine*, 2022.
* [Generative adversarial networks in medical image segmentation: a review](https://doi.org/10.1016/j.compbiomed.2021.105063) — Xun et al., *Computers in Biology and Medicine*, 2022.
* [Systematic review of GANs for medical image classification and segmentation](https://link.springer.com/article/10.1007/s10278-021-00556-w) — Jeong et al., *Journal of Digital Imaging*, 2022.
* [A survey on deep learning applied to medical images: from simple artificial neural networks to generative models](https://link.springer.com/article/10.1007/s00521-022-07953-4) — Celard et al., *Neural Computing and Applications*, 2022.
* [Generative adversarial networks in computer vision: a survey and taxonomy](https://arxiv.org/abs/1906.01529) — Wang, She & Ward, 2019.
* [Diffusion models in medical imaging: a comprehensive survey](https://doi.org/10.1016/j.media.2023.102846) — Kazerouni et al., *Medical Image Analysis*, 2023.
* [Diffusion models in vision: a survey](https://arxiv.org/abs/2209.04747) — Croitoru et al., *IEEE TPAMI*, 2023.

---

**Next:** pre-train on everything, adapt to anything.
→ [Episode 8: Foundation Models for Medical Imaging](08-foundation-models.md)
