# Training models: tips and tricks


## References

* For more general tips and tricks around model training (general because it isn't Medical Imaging in particular), [Andrei Karpathy's recipe from 2019](https://karpathy.github.io/2019/04/25/recipe/)is highly recommended. The contents of this episode is an extension of this with focus on medical image data sets. 

* [This](https://medium.com/miccai-educational-initiative/project-roadmap-for-the-medical-imaging-student-working-with-deep-learning-351add6066cf) blog post for more great tips while training models. This was a previous entry to the MICCAI Education Challenge as well!

* If you're inclined to use MONAI, consider following [this](https://github.com/Project-MONAI/tutorials/blob/main/2d_classification/mednist_tutorial.ipynb) tutorial. It follows an older version of the medMNIST data set, and uses MONAI to load the data and build models more easily.

* [MMAR](https://docs.nvidia.com/clara/clara-train-sdk/pt/mmar.html) is a Medical Model Archive designed by NVIDIA to organize all artifacts produced during the model development life cycle. This may be too heavyweight for research prototypes that one may begin with, but as things become more stable, such standards (much like BIDS for data storage) may help avoid further pains down the line.

* Consider following the [MICCAI Hackathon reproducibility checklist](https://github.com/JunMa11/MICCAI-Reproducibility-Checklist) to ensure that your pipeline is not too exotic, and future researchers can build on your work!
