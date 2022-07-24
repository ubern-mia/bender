# Training models: tips and tricks

At this stage, you hopefully have the data organized and curated in a way that it is ready to be used to train a deep learning model. In this episode of the [BENDER Series](https://github.com/ubern-mia/bender), we go over some tips and tricks to make the process of training, and more importantly logging metrics and intermediate results in a manageable manner. 

We start with the notebook [Dermamnist Initial Version](/episode03/dermamnist_v1_initial.ipynb) to demonstrate our recommended process. If you prefer a python script, consider opening [this](/episode03/dermamnist_v1_initial.py). This is admittedly a first attempt, and there are many improvements we can make. We invite you to make a copy of the notebook, and then make changes to it (you could simply copy changes from the scripts we point to) as we make progress in the versions below.

## Observations on the initial version:

General observations about how the code is organized, and things to keep in mind while running this.

![Model is not really learning all categories](/episode03/dermamnist_v1_initial/per_class_metrics.png)

Note from the image above that only the melanocytic nevi category is being learnt by the model, and since it is has the largest representation in both the training and validation/test set, the weighted average accuracy is quite high even though all the other categories have 0 accuracy. 

![training loss for initial version](/episode03/dermamnist_v1_initial/train_loss.png)

Training loss for initial version: seems to reduce with increasing iterations, but then flattens out. When it flattens out, it is really not very useful to train for more iterations, as the accuracy also flattens out. We will see in the third version, how this wasteful training could be avoided using 'validation patience'.

![validation accuracy for initial version](/episode03/dermamnist_v1_initial/val_acc.png)

Validation accuracy for initial version: even if we stopped at 10000 iterations, we would end up with the same validation accuracy as we do after 80000 iterations.

## Second version:

Modifying the momentum hyperparameter, keeping everything else the same. The difference between [v1](/episode03/dermamnist_v1_initial.py) and [v2](/episode03/dermamnist_v2_momentum0p9.py) is just the following line:

    optimizer = torch.optim.SGD(model.parameters(), lr=0.000005, momentum=0.5)

changed to 

    optimizer = torch.optim.SGD(model.parameters(), lr=0.000005, momentum=0.9)

Make this change in your notebook, and verify that you see similar results as we do, in the training loss and validation accuracy.

![training loss after momemtum change](/episode03/dermamnist_v2_momentum0p9/train_loss.png)

Training loss after change of momentum to 0.9 from 0.5

![validation accuracy after momemtum change](/episode03/dermamnist_v2_momentum0p9/val_acc.png)

Validation accuracy after change of momentum to 0.9 from 0.5

## Third version:

Changing the learning rate, and also introducing validation patience, to avoid wasting training cycles if the model is not generalizing.

We change the learning rate on this line:

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.5)

changed to 

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

![training loss after change in learning rate](/episode03/dermamnist_v3_lr0p005_val_patience/train_loss.png)

Training loss after change of learning rate to 0.005 from 0.000005

![validation accuracy after momemtum change](/episode03/dermamnist_v3_lr0p005_val_patience/val_acc.png)

Validation accuracy after change of learning rate to 0.005 from 0.000005

## Fourth version:

Switching to the ADAM optimizer, while also introducing tensorboard for better organized logging of metrics while training. 

## Fifth version:

Modifying the network itself to be deeper and hopefully more capable in handling a difficult data distribution.

## Sixth version:

Making the network even more deeper - based on improvements seen in the previous version.

## Seventh version:

Adding data augmentation at the end, to imclude some regularization and possibly improve generalization.

## References

* For more general tips and tricks around model training (general because it isn't Medical Imaging in particular), [Andrei Karpathy's recipe from 2019](https://karpathy.github.io/2019/04/25/recipe/)is highly recommended. The contents of this episode is an extension of this with focus on medical image data sets. 

* [This](https://medium.com/miccai-educational-initiative/project-roadmap-for-the-medical-imaging-student-working-with-deep-learning-351add6066cf) blog post for more great tips while training models. This was a previous entry to the MICCAI Education Challenge as well!

* If you're inclined to use MONAI, consider following [this](https://github.com/Project-MONAI/tutorials/blob/main/2d_classification/mednist_tutorial.ipynb) tutorial. It follows an older version of the medMNIST data set, and uses MONAI to load the data and build models more easily.

* [MMAR](https://docs.nvidia.com/clara/clara-train-sdk/pt/mmar.html) is a Medical Model Archive designed by NVIDIA to organize all artifacts produced during the model development life cycle. This may be too heavyweight for research prototypes that one may begin with, but as things become more stable, such standards (much like BIDS for data storage) may help avoid further pains down the line.

* Consider following the [MICCAI Hackathon reproducibility checklist](https://github.com/JunMa11/MICCAI-Reproducibility-Checklist) to ensure that your pipeline is not too exotic, and future researchers can build on your work!
