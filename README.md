# The 🤖 BENDER series:

Are you a new(ish) Graduate or a super-enthusiastic Undergraduate student working with Medical Image data, and pondering over how to get that Deep Learning model to train with it? If so, this repository is for you: 

📖 **Full notes, episode-by-episode, are on the companion website: [ubern-mia.github.io/bender](https://ubern-mia.github.io/bender/)**

It supports the [BENDER (BEst practices in medical imagiNg DEep leaRning) series of videos](https://www.youtube.com/playlist?list=PLFwdflE4leRpqIz-F68pvwFATIOEwrSHp), which is submitted to [the MICCAI Education Challenge, 2022](https://miccai-sb.github.io/challenge.html).

We invite you to watch along the following episodes (click on the pictures below to watch on YouTube) that track the life and times of a new student who has just started working with this kind of data: through the ups and downs of the journey to build a State-of-The-Art model. 

We hope through the experience of this student, you learn not to make the same mistakes, and get a head-start in your own exploration.  

--------------------

## Episode 1: “Exploratory analysis: The Data Pile” 

<a href="http://www.youtube.com/watch?feature=player_embedded&v=NtszpkE0gc4
" target="_blank"><img src="http://img.youtube.com/vi/NtszpkE0gc4/0.jpg" 
alt="BENDER episode 01" width="640" height="480" border="10" /></a>

📖 [Read the full notes for this episode](https://ubern-mia.github.io/bender/episodes/01-exploratory-data-analysis/)

See [checklist](/exploratory-data-analysis/checklist.md) for a list of keepawakes while dealing with clinical data. We believe checking off all of these as a bare minimum would help avoid 💣 surprises later on. 

A downloadable PDF version is [here](/exploratory-data-analysis/checklist.pdf).

[Exploring the Dermamnist data set](/exploratory-data-analysis/explore_dermamnist.ipynb) is a notebook that uses data from  [MedMNIST](https://medmnist.com/) to demonstrate examples of the kinds of details you should look for, while analyzing your clinical data.

If you are lucky (or not 😉) to be working with DICOM data, [Exploring DICOM tags](/exploratory-data-analysis/explore_dicom.ipynb) is a notebook to demonstrate how to go about looking at the non-image metadata as well! 

[Here](http://www.r2d3.us/) is a wonderful (non-medical-imaging) example of how simply "looking" at the data prior to modeling helps make sense of the landscape, and is a step that one must not neglect while starting off!

For more specialized data, there are standards like [BIDS](https://bids.neuroimaging.io), Brain Imaging Data structure, for organizing multiple subject files for neuroimaging. Consider conforming to such formats to ensure easier read/write/convert workflows with your data set.

In this context, Data-centered AI is gaining more attention, especially through these papers:
- [DataPerf: Benchmarks for Data-Centric AI Development](https://arxiv.org/abs/2207.10062)
- [Advances, challenges and opportunities in creating data for trustworthy AI](https://www.nature.com/articles/s42256-022-00516-1)
- [IEEE Spectrum article: Andrew Ng, AI Minimalist: The Machine-Learning Pioneer Says Small is the New Big](https://ieeexplore.ieee.org/document/9754503)

--------------------

## Episode 2: “Terminology: Meet the Experts” 

<a href="http://www.youtube.com/watch?feature=player_embedded&v=jGLBcMyiehg
" target="_blank"><img src="http://img.youtube.com/vi/jGLBcMyiehg/0.jpg" 
alt="BENDER episode 02" width="640" height="480" border="10" /></a>

📖 [Read the full notes for this episode](https://ubern-mia.github.io/bender/episodes/02-terminology/)

Click [here for a glossary](/terminology-meet-experts/glossar.md) of common terms used in the Deep Learning world, with a special focus on Medical Imaging and clinical lingo. 

A downloadable PDF version is [here](/terminology-meet-experts/glossar.pdf).

--------------------

## Episode 3: “Training models: Good model training shall you strive for”

<a href="http://www.youtube.com/watch?feature=player_embedded&v=f0wd8EvRiH0
" target="_blank"><img src="http://img.youtube.com/vi/f0wd8EvRiH0/0.jpg" 
alt="BENDER episode 03" width="640" height="480" border="10" /></a>

📖 [Read the full notes for this episode](https://ubern-mia.github.io/bender/episodes/03-training-models/)

[Follow along this episode](/training-models/README.md) for the next step of actually building a model.

We start with a naive implementation (as [ipynb](/training-models/dermamnist_v1_initial.ipynb) or a [py script](/training-models/dermamnist_v1_initial.py)), walking through the process of improving it iteratively in seven versions to finally beat the [benchmark listed on the MedMNIST webpage](https://medmnist.com) in an organized experimental fashion!

--------------------

## Episode 4: “Evaluating your model: Born to Deploy”

<a href="http://www.youtube.com/watch?feature=player_embedded&v=YwM7qwqSy9k
" target="_blank"><img src="http://img.youtube.com/vi/YwM7qwqSy9k/0.jpg" 
alt="BENDER episode 04" width="640" height="480" border="10" /></a>

📖 [Read the full notes for this episode](https://ubern-mia.github.io/bender/episodes/04-evaluation-and-deployment/)

In [this episode](/evaluating-and-deploying-model/README.md), the focus is on evaluation and deployment: specific points that are important to keep in mind for clinical relevance, out-of-distribution data, and other interesting bits.

--------------------

## Episode 5: “Becoming one with the gradients”

<a href="http://www.youtube.com/watch?feature=player_embedded&v=hr1szGBP7Ps
" target="_blank"><img src="http://img.youtube.com/vi/hr1szGBP7Ps/0.jpg" 
alt="BENDER episode 05" width="640" height="480" border="10" /></a>

📖 [Read the full notes for this episode](https://ubern-mia.github.io/bender/episodes/05-interpretability/)

[Follow along this episode](/gradient-based-interpretability/README.md) for the next step in understanding gradients and their role in deep learning interpretability.

--------------------

## Episode 6: “The U-Net Model”

<a href="http://www.youtube.com/watch?feature=player_embedded&v=AuDio_Clxo8
" target="_blank"><img src="http://img.youtube.com/vi/AuDio_Clxo8/0.jpg" 
alt="BENDER episode 06" width="640" height="480" border="10" /></a>

📖 [Read the full notes for this episode](https://ubern-mia.github.io/bender/episodes/06-u-net/)

[Follow along this episode](/u-net-model/README.md) for an introduction to the U-Net model, a popular architecture in medical image segmentation.

--------------------

## Episode 7: “Generative Models in Medical Imaging”

<a href="http://www.youtube.com/watch?feature=player_embedded&v=Bp3OUSdtkfY
" target="_blank"><img src="http://img.youtube.com/vi/Bp3OUSdtkfY/0.jpg" 
alt="BENDER episode 07" width="640" height="480" border="10" /></a>

📖 [Read the full notes for this episode](https://ubern-mia.github.io/bender/episodes/07-generative-models/)

[Follow along this episode](/generative-models/README.md) for insights into generative models and their applications in medical imaging.

--------------------

## Episode 8: “Foundation Models for Medical Imaging”

<a href="http://www.youtube.com/watch?feature=player_embedded&v=JVfEAjbw5hk
" target="_blank"><img src="http://img.youtube.com/vi/JVfEAjbw5hk/0.jpg" 
alt="BENDER episode 08" width="640" height="480" border="10" /></a>

📖 [Read the full notes for this episode](https://ubern-mia.github.io/bender/episodes/08-foundation-models/)

[Follow along this episode](/foundation-models/README.md) for a discussion on foundation models and their potential in medical imaging.

--------------------

## Episode 9: “Federated Learning in Medical Imaging”

<a href="http://www.youtube.com/watch?feature=player_embedded&v=i2kaDRe8BDo
" target="_blank"><img src="http://img.youtube.com/vi/i2kaDRe8BDo/0.jpg" 
alt="BENDER episode 09" width="640" height="480" border="10" /></a>

📖 [Read the full notes for this episode](https://ubern-mia.github.io/bender/episodes/09-federated-learning/)

In [this episode](/federated-learning/README.md), Satish and Min collaborate across the world to train a model together — without any of their data leaving the hospital. We cover how the federated loop works, why non-IID hospital data breaks it, why “the data never leaves” does not by itself mean “private”, and which frameworks to start with.

To try it yourself, [dermamnist_federated.py](/federated-learning/dermamnist_federated.py) simulates a five-hospital federation over the same data and model as episode 3 (`make run-fed`), with the local-only and pooled baselines you need to judge whether federating was worth it.

--------------------

We hope you learn something new about how to get started with Medical Imaging and Deep Learning, and more importantly, that you have fun while learning! (If you have suggestions for improvement, please do not hesitate to create an [issue here](https://github.com/ubern-mia/bender/issues))

👋 The Medical Imaging Analysis group at Universität Bern