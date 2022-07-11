# The 🤖 BENDER series:

Are you a new(ish) Graduate or a super-enthusiastic Undergraduate student working with Medical Image data, and pondering over how to get that Deep Learning model to train with it? If so, this repository is for you: it supports the [BENDER (BEst practices in medical imagiNg DEep leaRning) series of videos](https://www.youtube.com/playlist?list=PLFwdflE4leRpqIz-F68pvwFATIOEwrSHp), which is submitted to [the MICCAI Education Challenge, 2022](https://miccai-sb.github.io/challenge.html).

We would invite you to watch along the following four episodes that track the life and times of a new student who has just started working with this kind of data: through the ups and downs of his journey to build a State-of-The-Art model. We hope through the experience of this student, you learn not to make the same mistakes, and get a head-start in your own exploration.  

--------------------

## Episode 1: “The data pile” 

<a href="http://www.youtube.com/watch?feature=player_embedded&v=NtszpkE0gc4
" target="_blank"><img src="http://img.youtube.com/vi/NtszpkE0gc4/0.jpg" 
alt="BENDER episode 01" width="640" height="480" border="10" /></a>

See [checklist](/episode01/checklist.md) for a list of keepawakes while dealing with clinical data. We believe checking off all of these as a bare minimum would help avoid 💣 surprises later on.

[Exploring the Dermamnist data set](https://github.com/ubern-mia/bender/blob/main/episode01/explore_dermamnist.ipynb) is a notebook that uses data from  [MedMNIST](https://medmnist.com/) to demonstrate examples of the kinds of details you should look for, while analyzing your clinical data.

If you are lucky (or not 😉) to be working with DICOM data, [Exploring DICOM tags](https://github.com/ubern-mia/bender/blob/main/episode01/explore_dicom.ipynb) is a notebook to demonstrate how to go about looking at the non-image metadata as well! 

[Here](http://www.r2d3.us/) is a wonderful (non-medical-imaging) example of how simply "looking" at the data prior to modeling helps make sense of the landscape, and is a step that one must not neglect while starting off!

--------------------

## Episode 2: “Medical image computing is not the same as computing on medical images” 

<a href="http://www.youtube.com/watch?feature=player_embedded&v=jGLBcMyiehg
" target="_blank"><img src="http://img.youtube.com/vi/jGLBcMyiehg/0.jpg" 
alt="BENDER episode 02" width="640" height="480" border="10" /></a>

See [glossary](/episode02/glossar.md) for a glossary of common terms used in the Deep Learning world, with a special focus on Medical Imaging and clinical lingo.

--------------------

## Episode 3: “Modeling”

<a href="http://www.youtube.com/watch?feature=player_embedded&v=f0wd8EvRiH0
" target="_blank"><img src="http://img.youtube.com/vi/f0wd8EvRiH0/0.jpg" 
alt="BENDER episode 03" width="640" height="480" border="10" /></a>

This episode focuses on the next step of actually building a model: starting from a naive implementation, walking through the process of improving it iteratively to get to a better state, while also understanding how to evaluate if it is indeed getting better!

--------------------

## Episode 4: “Evaluating your model: we are out of the sandbox”

<a href="http://www.youtube.com/watch?feature=player_embedded&v=YwM7qwqSy9k
" target="_blank"><img src="http://img.youtube.com/vi/YwM7qwqSy9k/0.jpg" 
alt="BENDER episode 04" width="640" height="480" border="10" /></a>

In this final episode, the focus is on evaluation and deployment: specific points that are important to keep in mind for clinical relevance, out-of-distribution data, and other interesting bits.

We hope you learn something new about how to get started with Medical Imaging and Deep Learning, and more importantly, that you have fun while learning! (If you have suggestions for improvement, please do not hesitate to create an [issue here](https://github.com/ubern-mia/bender/issues))


👋 The Medical Imaging Analysis group at Universität Bern
