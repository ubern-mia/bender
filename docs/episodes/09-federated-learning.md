# Episode 9 — Federated Learning in Medical Imaging

<div class="video-embed">
  <iframe src="https://www.youtube-nocookie.com/embed/i2kaDRe8BDo" title="BENDER: Federated Learning in Medical Imaging" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

<div class="episode-meta">
  <span><strong>Runtime</strong> 7:43</span>
  <span><strong>Published</strong> 17 December 2025</span>
  <span><strong>Characters</strong> Satish and Min</span>
  <span><strong>Companion</strong> <code>dermamnist_federated.py</code> · <code>make run-fed</code></span>
</div>

## What happens in the episode

Satish and Min want to build their next model together — except their data sits in two
hospitals, on two continents, under two very different sets of rules. Nobody is allowed to
ship the images anywhere.

So instead of moving the data to the model, they move the model to the data. That is
federated learning, and this episode is about the parts of it that surprise people.

By now you have curated data ([episode 1](01-exploratory-data-analysis.md)), trained a model
in an organized way ([episode 3](03-training-models.md)), and thought hard about how it will
behave in the clinic ([episode 4](04-evaluation-and-deployment.md)). Federated learning does
not replace any of that. It makes every one of those steps harder.

## Why bother? The data usually cannot travel

In medical imaging the bottleneck is rarely the GPU: it is permission. Patient data is
governed by regulation (GDPR, HIPAA and their local cousins), by the ethics approval under
which it was collected, and often simply by the fact that a data transfer agreement between
two institutions takes longer to negotiate than the project itself.

Federated learning flips the problem around. The images stay inside each hospital's
firewall, and only model parameters (or updates to them) travel. This lets a rare-disease
cohort of 40 subjects in Bern be trained together with 400 in Singapore, without either site
handing over a single voxel.

## The loop itself is almost boringly simple

1. A central server initializes a model and broadcasts the weights to every participating site.
2. Each site trains on its own local data for a few epochs — **this is exactly the training loop from [episode 3](03-training-models.md), unchanged**.
3. Each site sends back only its updated weights (or the delta), never the data.
4. The server aggregates these, typically as a weighted average by number of local samples, and broadcasts the new global model.
5. Repeat for a few hundred rounds.

That is genuinely the whole algorithm — [FedAvg](https://arxiv.org/abs/1602.05629), McMahan
et al., 2017. Everything that makes federated learning hard lives in the assumptions this
loop quietly makes, and in medical imaging most of those assumptions are false.

## Non-IID data is the normal case, not the edge case

FedAvg works beautifully when every client's data looks like a random sample from one
distribution. Hospitals are the opposite of that:

* different scanner vendors, field strengths and reconstruction kernels — the same
  single-source bias from [episode 4](04-evaluation-and-deployment.md), now baked in per
  client;
* different acquisition protocols and slice thicknesses;
* wildly different label distributions — a tertiary referral centre sees the rare classes, a
  district hospital sees the common ones;
* different annotators, and therefore different label noise.

Averaged naively, clients pull the global model in conflicting directions and it converges
slowly, or to something worse than any single site could have trained alone. Two well-known
mitigations:

* **[FedProx](https://arxiv.org/abs/1812.06127)** adds a proximal term that keeps local
  updates from drifting too far from the global model.
* **[FedBN](https://arxiv.org/abs/2102.07623)** simply keeps batch-normalization statistics
  local and never averages them, which absorbs a surprising amount of the scanner-to-scanner
  shift.

!!! tip "Always compare against two baselines"
    Before declaring victory: the model each site could have trained **alone**, and the model
    you would get if the data *could* be **pooled**. Federated learning is worth it when you
    land closer to the second than the first. The companion script gives you both.

## "The data never leaves" is not the same as "this is private"

This is the point most worth taking away.

Gradients are derived from data, and a determined server can work backwards from them.
[Deep Leakage from Gradients](https://arxiv.org/abs/1906.08935) and
[Inverting Gradients](https://arxiv.org/abs/2003.14053) both reconstruct recognizable
training images from shared updates alone. For medical images that is a re-identification
risk, not a theoretical curiosity.

If your project makes a privacy claim to an ethics board, it needs to rest on something
stronger than the architecture diagram:

* **Secure aggregation** ([Bonawitz et al.](https://eprint.iacr.org/2017/281)) lets the
  server compute the sum of client updates without seeing any individual one.
* **Differential privacy** adds calibrated noise with a stated privacy budget;
  [Opacus](https://opacus.ai) is a practical starting point in PyTorch. Expect to pay for it
  in accuracy, and report that cost honestly.
* **Threat model first.** Decide explicitly whether you trust the server, the other clients,
  and the network. The defenses you need follow from that answer, not the other way round.

## It has actually been done, at scale

* **[EXAM](https://www.nature.com/articles/s41591-021-01506-3)** (Dayan et al., *Nature
  Medicine*, 2021) — 20 institutions worldwide predicting oxygen requirements in COVID-19
  patients, with generalization gains of around 16% over locally trained models.
* **[FeTS](https://www.nature.com/articles/s41467-022-33407-5)** (Pati et al., *Nature
  Communications*, 2022) — the Federated Tumor Segmentation initiative, 71 sites, the largest
  glioblastoma segmentation study to date. The [FeTS tooling](https://github.com/FETS-AI/Front-End)
  is public.
* **[Swarm Learning](https://www.nature.com/articles/s41586-021-03583-3)** (Warnat-Herresthal
  et al., *Nature*, 2021) — a decentralized variant with no central server at all.

## Frameworks: please do not write your own

The interesting research questions are in aggregation and privacy, not in TLS handshakes and
client orchestration.

| Framework | Why you might pick it |
|---|---|
| [NVIDIA FLARE](https://github.com/NVIDIA/NVFlare) | Production-oriented, strong medical imaging track record, integrates with [MONAI](https://github.com/Project-MONAI/MONAI) |
| [Flower](https://flower.ai) | Framework-agnostic, easy to simulate many clients on one machine — the best place to prototype, and what the companion script uses |
| [OpenFL](https://github.com/securefederatedai/openfl) | Originally from Intel and the FeTS work, now under the Linux Foundation |
| [Fed-BioMed](https://fedbiomed.org) | Built specifically around biomedical and hospital deployments |

A useful trick: every one of these can *simulate* a federation on your laptop by
partitioning a public dataset across virtual clients. Do that first — you will have learned
most of the practical lessons before a single IT department is involved.

So let us do exactly that.

## Run a federation on your own laptop

[`dermamnist_federated.py`](https://github.com/ubern-mia/bender/blob/main/federated-learning/dermamnist_federated.py)
simulates a five-hospital federation over the same DermaMNIST data and the same CNN as
[episode 3](03-training-models.md), using [Flower](https://flower.ai). The only thing that
changes between the two episodes is *where the data lives*.

```bash
make install           # installs flwr[simulation] along with everything else

make run-fed-partition # just show how the data was split across hospitals
make run-fed           # train the federation (FedAvg, 5 clients, 10 rounds)
make run-fed-local     # baseline 1: each hospital training alone
make run-fed-central   # baseline 2: the pooled upper bound
```

Pass flags through `FED_ARGS`:

```bash
make run-fed FED_ARGS="--alpha 0.1"                    # far more label skew
make run-fed FED_ARGS="--strategy fedprox --alpha 0.1" # does FedProx rescue it?
make run-fed FED_ARGS="--fedbn"                        # keep BatchNorm local
make run-fed FED_ARGS="--num-clients 20 --num-rounds 30"
```

| Flag | What it does | Default |
|---|---|---|
| `--num-clients` | How many hospitals to simulate | 5 |
| `--alpha` | Dirichlet concentration for the split. Lower = more label skew; `--alpha 100` is effectively IID | 0.5 |
| `--num-rounds` | Communication rounds | 10 |
| `--local-epochs` | How much local training happens inside each round | 1 |
| `--strategy` | `fedavg` or `fedprox` (with `--proximal-mu`) | `fedavg` |
| `--fedbn` | Keep BatchNorm parameters and running statistics at each client | off |
| `--batch-size` / `--learning-rate` | As in episode 3 | 64 / 0.005 |
| `--seed` | Reproducibility | 42 |

### What to actually look at

**First, the split.** Every mode prints the per-client class histogram before training, and
it is worth staring at:

```
  client      n  actinic ke  basal cell  benign ker  dermatofib    melanoma  melanocyti  vascular l
       0   2002          32         105         275           4           5        1523          58
       1   1136          54          81         128           1         162         710           0
       2   2221         132          70          67          21         379        1539          13
       3   1437           0          99         298          45         232         737          26
       4    211          10           4           1           9           1         184           2
```

Client 3 has never seen an actinic keratosis. Client 1 has never seen a vascular lesion.
Client 4 has 211 images in total against client 2's 2221. Nobody designed this to be
awkward — it is one draw from a Dirichlet with `alpha=0.5`, and it is a *mild* version of
what real multi-site studies look like.

**Second, accuracy next to balanced accuracy.** The script reports both every round, and
early on you will see something like `accuracy 0.6688, balanced 0.1429`. That is not a model
that has learned something: 67% of DermaMNIST is melanocytic nevi, and 0.1429 is exactly 1/7
— the model has collapsed to predicting the majority class for everything. This is
[episode 4's](04-evaluation-and-deployment.md) warning about average metrics, showing up in
the very first thing you run. Watch whether balanced accuracy climbs off the floor: that,
not accuracy, is the interesting curve.

**Third, the comparison that decides the question.** Run all three modes and put the numbers
side by side. If `federated` beats `local`, the collaboration bought something. If it lands
near `centralized`, it bought nearly everything pooling would have. If it lands below the
best local model, something is wrong — usually the skew, and that is where
`--strategy fedprox` and `--fedbn` earn their keep.

### What we got, running exactly the commands above

Five clients, `alpha=0.5`, ten rounds of one local epoch each, one seed, on a laptop CPU:

| Setting | Accuracy | Balanced accuracy |
|---|---|---|
| Local only (mean over the 5 hospitals) | 0.6568 | 0.2110 |
| **Federated** (FedAvg, 10 rounds) | 0.6599 | **0.2556** |
| Centralized (data pooled) | 0.6603 | 0.3164 |

Look at the accuracy column first: 0.657, 0.660, 0.660. Three genuinely different models —
one trained alone on 1400 images, one trained across five hospitals, one trained on
everything — and plain accuracy cannot tell them apart at all, because all three are mostly
reporting how common melanocytic nevi are.

Now look at balanced accuracy, and the ordering appears exactly where theory says it should:
federating beat training alone (0.256 vs 0.211), and recovered a bit more than half the gap
to what pooling the data would have achieved (0.316). That is the whole value proposition of
federated learning, in one row of a table — and it was invisible in the metric most papers
lead with.

![FedAvg over 10 rounds, five clients, alpha 0.5](../figures/federated-learning/dermamnist_federated/fedavg_alpha0.5_clients5.png)

The plot the script saves tells the same story: accuracy jumps to 0.67 in a single round and
then flatlines, while balanced accuracy grinds slowly upward. The blue curve is the model
learning the class prior. The orange curve is the model learning dermatology.

!!! note "Two honest caveats"
    This is one seed and about ten epochs of training, so the numbers are noisy and none of
    these models is any good yet; and DermaMNIST split by Dirichlet is a friendly imitation
    of a real federation, not a substitute for one. Re-run with `--alpha 0.1` to watch the
    gap widen, then see whether `--strategy fedprox` closes it.

**A note on `--fedbn`:** when BatchNorm parameters stay local there is no complete global
model on the server to evaluate, so the script switches to asking each client to evaluate
and averaging the results. That is not a workaround, it is the trade-off FedBN actually
makes — worth knowing before you promise someone a single downloadable model at the end of
the project.

**One caveat on the implementation:** the script drives Flower through `run_simulation()`,
which keeps the entire example in one readable file. Flower now marks that entry point as
deprecated in favour of the `flwr run` CLI and its project layout — for a real deployment,
follow [their tutorial](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)
rather than this file's structure.

## Before you federate: a short checklist

- [ ] Is federated learning actually needed, or would a data transfer agreement be faster? (Sometimes it would.)
- [ ] Have you simulated the federation locally, with a realistic non-IID split?
- [ ] Do you have baselines for local-only and (where possible) centrally pooled training?
- [ ] Is preprocessing identical at every site, and *verified* — not merely documented?
- [ ] Do all sites agree on the label definitions and the annotation protocol? (See [episode 2](02-terminology.md).)
- [ ] Who validates the global model, on which held-out data, and at which site?
- [ ] What is the threat model, and which defense addresses it?
- [ ] Who owns the resulting model, and what happens to it when the project ends?
- [ ] What happens when a site drops out mid-training, or its connection dies at round 300?

The last one is not a joke: in real deployments, stragglers and dropouts cost more project
time than the algorithm ever does.

## References and further reading

* **[Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)** — McMahan et al., AISTATS 2017. The FedAvg paper.
* **[The future of digital health with federated learning](https://www.nature.com/articles/s41746-020-00323-1)** — Rieke et al., *npj Digital Medicine*, 2020. The medical-imaging case, made well.
* **[Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977)** — Kairouz et al., 2019. The reference that catalogues what the simple loop assumes.
* **[FedProx: Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)** — Li et al., MLSys 2020.
* **[FedBN: Federated Learning on Non-IID Features via Local Batch Normalization](https://arxiv.org/abs/2102.07623)** — Li et al., ICLR 2021.
* **[Deep Leakage from Gradients](https://arxiv.org/abs/1906.08935)** — Zhu, Liu & Han, NeurIPS 2019.
* **[Inverting Gradients — How easy is it to break privacy in federated learning?](https://arxiv.org/abs/2003.14053)** — Geiping et al., NeurIPS 2020.
* **[Practical Secure Aggregation for Privacy-Preserving Machine Learning](https://eprint.iacr.org/2017/281)** — Bonawitz et al., CCS 2017.
* **[Federated learning for predicting clinical outcomes in patients with COVID-19 (EXAM)](https://www.nature.com/articles/s41591-021-01506-3)** — Dayan et al., *Nature Medicine*, 2021.
* **[Federated learning enables big data for rare cancer boundary detection (FeTS)](https://www.nature.com/articles/s41467-022-33407-5)** — Pati et al., *Nature Communications*, 2022.
* **[Swarm Learning for decentralized and confidential clinical machine learning](https://www.nature.com/articles/s41586-021-03583-3)** — Warnat-Herresthal et al., *Nature*, 2021.
* **[Opacus](https://opacus.ai)** — differential privacy in PyTorch, for when the ethics board asks.

---

That is the series so far. If you have followed all nine episodes: you have curated data,
trained and improved a model with discipline, evaluated it honestly, opened it up, met the
architectures, generated new images, considered a foundation model, and trained across
hospitals that could not share anything.

Please [open an issue](https://github.com/ubern-mia/bender/issues) if you would like to see a
topic covered next. → [Back to all episodes](index.md)
