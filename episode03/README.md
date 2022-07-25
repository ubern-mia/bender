# Training models: tips and tricks

At this stage, you hopefully have the data organized and curated in a way that it is ready to be used to train a deep learning model. In this episode of the [BENDER Series](https://github.com/ubern-mia/bender), we go over some tips and tricks to make the process of training, and more importantly logging metrics and intermediate results in a manageable manner. 

--------------------

## Initial version:

We start with the notebook [DermaMNIST Initial Version](/episode03/dermamnist_v1_initial.ipynb) to demonstrate our recommended process. If you prefer a python script, consider opening [this](/episode03/dermamnist_v1_initial.py). This is admittedly a first attempt, and there are many improvements we can make. 

We invite you to make a copy of the notebook, and then make changes to it (you could simply copy changes from the scripts we point to) as we make progress in the versions below.

![Model is not really learning all categories](/episode03/dermamnist_v1_initial/per_class_metrics.png)

Note from the image above that only the `melanocytic nevi` category is being learnt by the model, and since it is has the largest representation in both the training and validation/test set, the weighted average accuracy is quite high even though all the other categories have 0 accuracy. 

![training loss for initial version](/episode03/dermamnist_v1_initial/train_loss.png)

Training loss for initial version: seems to reduce with increasing iterations, but then flattens out. When it flattens out, it is really not very useful to train for more iterations, as the accuracy also flattens out. We will see in the third version, how this wasteful training could be avoided using 'validation patience'.

![validation accuracy for initial version](/episode03/dermamnist_v1_initial/val_acc.png)

Validation accuracy for initial version: even if we stopped at 10000 iterations, we would end up with the same validation accuracy as we do after 80000 iterations.

--------------------

## Second version:

For this version, we modify only the momentum hyperparameter, keeping everything else the same. The difference between [v1](/episode03/dermamnist_v1_initial.py) and [v2](/episode03/dermamnist_v2_momentum0p9.py) is just the following line:

    optimizer = torch.optim.SGD(model.parameters(), lr=0.000005, momentum=0.5)

changed to 

    optimizer = torch.optim.SGD(model.parameters(), lr=0.000005, momentum=0.9)

Make this change in your notebook, and verify that you see similar results as we do, in the training loss and validation accuracy.

![training loss after momemtum change](/episode03/dermamnist_v2_momentum0p9/train_loss.png)

Training loss after change of momentum to 0.9 from 0.5: note that this is still noisy like v1, but the loss reduces faster: breaching the value of 1.0 is under 10000 iterations as compared to 20000 earlier. 

![validation accuracy after momemtum change](/episode03/dermamnist_v2_momentum0p9/val_acc.png)

Validation accuracy after change of momentum to 0.9 from 0.5: there's not much change here, the accuracy appears to remain in the same range, indicating that we should explore modifying other hyperparameters, or, even changing the model itself.

--------------------

## Third version:

In the spirit of tweaking hyperparameters, we make another change here, where we increase the learning rate (one of the most sensitive, and hence most tweaked hyperparameters), so that the network weights can learn 'faster'. The functional difference between [v2](/episode03/dermamnist_v2_momentum0p9.py) and [v3](/episode03/dermamnist_v3_lr0p005_val_patience.py) is just the following line:

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.5)

changed to 

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

Additionally, we make another modification which doesn't change the training algorithm per-se, but adds a level of verification if the training is going well or not. Each time we evaluate the validation accuracy, we track if it is higher or lower than the previous best. If for a fixed number of evaluations, the accuracy is lower or equal to the previous best, we believe then that the network is not learning anything new, and so we stop training. This fixed number is called the 'validation patience' and it helps avoid situations where the validation accuracy is flat (like in the two versions above), or worse, if the network overfits (where the validation accuracy may reduce).

To make this change in your notebook, look for the `patience` parameter in [this](/episode03/dermamnist_v3_lr0p005_val_patience.py) script, and include all the lines that contain it in your training function.

![training loss after change in learning rate](/episode03/dermamnist_v3_lr0p005_val_patience/train_loss.png)

Training loss after change of learning rate to 0.005 from 0.000005. Note that the training loss appears to be very noisy, but is lower than the previous two versions.

![validation accuracy after momemtum change](/episode03/dermamnist_v3_lr0p005_val_patience/val_acc.png)

Validation accuracy after change of learning rate to 0.005 from 0.000005. This shows a significant improvement already compared to the previous two versions: the accuracies breach the 0.75 level, and the hope is that the test accuracies are similar as well (so that we verify the generalization capability and not overfit on the validation data ;-)).

--------------------

## Fourth version:

For [this version (v4)](/episode03/dermamnist_v4_adam_TB.py), we make a minor change in the optimizer: we switch from SGD (Stochastic Gradient Descent) to Adam (Adaptive Moment), which is known to be 'safer' (see [here](https://karpathy.github.io/2019/04/25/recipe/) for more) and less forgiving to hyperparameter variations.

Hence, functionally, the only change here is the following line:

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

changed to

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

However, we also introduce [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html) to log the training progress for the same parameters: training (and additionally validation) loss as well as validation (and additionally training) accuracy. 

Since the code changes are relatively large compared to the previous versions, we have prepared a new [ipynb](/episode03/dermamnist_v4_adam_TB.ipynb) notebook to start off again, if you prefer. Run this notebook on your own to verify that you see results that confirm improvement over the previous versions, and also browse through the Tensorboard logs (like shown in the videos) to get a better handle over the performance of this model.

![Training and Validation losses and accuracies](/episode03/dermamnist_v4_adam_TB/train_val_TB_plots.png)

Here are the training (left) and validation (right) accuracies (top) and losses (bottom) using Tensorboard. The curves are smoothed for better visualization (smoothing = 0.898). Note how the training loss keeps reducing and accuracy keeps rising, but the validation loss plateaus and even rises, while the accuracy plateaus and also falls slightly. Our code is setup to save the model with the highest validation accuracy, but what this indicates is that our model has overfit on the training data, and the next versions attempt to handle just that. 

![Test accuracy and classification report](/episode03/dermamnist_v4_adam_TB/test_accuracy_v4.png)

The test accuracy for our model is 0.762, and as you can see, all the categories (and not just melanocytic nevi) have a non-zero precision, which is an improvement over the first naive version! Also, 0.762 is already second highest in the [DermaMNIST benchmarks on the MedMNIST webpage](https://medmnist.com/)!

--------------------

## Fifth version:

In this version, to try and avoid overfitting to the training data, we increase the capacity of our model by adding more layers, with the intention of allowing the model to be more expressive and hence learn the nuances of the actual distribution of categories better (than simply memorizing the training data). 

To do so, [v5](/episode03/dermamnist_v5_deeper_network.py) includes just the following change in the model `init` method:

    ...
    nn.Conv2d(64, 128, (3, 3), padding=1, stride=2, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    # (64, 16, 16)
    nn.Conv2d(128, 128, (3, 3), padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    # (64, 16, 16)
    nn.Conv2d(128, 128, (3, 3), padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    # (128, 8, 8)
    ...

Our network now is 6 layers deep (each layer is assumed to mean convolution + ReLU + Batch Norm). Everything else is maintained to be the same, and we continue to use Tensorboard to log and compare the results.

![Training and Validation losses and accuracies](/episode03/dermamnist_v5_deeper_network/training_val_v4_v5_comparison.png)

In the image above, the blue curves correspond to [version 4](/episode03/dermamnist_v4_adam_TB.py), and gray curves are for the [current version](/episode03/dermamnist_v5_deeper_network.py). Note that the training as well as validation losses are lower with this deeper network, while the training accuracy has a higher slope, while the validation accuracy still tracks similar to [version 4](/episode03/dermamnist_v4_adam_TB.py). This means that we could make the model even more capable/powerful to try and understand this distribution.

![Test accuracy and classification report](/episode03/dermamnist_v5_deeper_network/test_accuracy_v5.png)

For this version, the test accuracy is in fact lower than the previous one, and is 0.755 (still competitive in the benchmark!). Note also that the dermatofibroma metrics are all 0, indicating that this model completely ignores this category, yet achieves good average results. This is a good reason to visualize the entire confusion chart to make sure such behavior does not occur! 

--------------------

## Sixth version:

This [version](/episode03/dermamnist_v6_even_deeper_network.py) changes only the network again, and makes it even more deeper, with 8 layers now. Everything else is left the same, so that we are able to investigate the impact of each of these individual changes. The code change is again only in the `init` method of the network definition, where the following lines are included (see full changes between [v5](/episode03/dermamnist_v5_deeper_network.py) and [v6](/episode03/dermamnist_v6_even_deeper_network.py)). 

    ...
    nn.ReLU(),
    # (64, 16, 16)
    nn.Conv2d(128, 256, (3, 3), padding=1, stride=2, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    # (128, 8, 8)
    nn.Conv2d(256, 256, (3, 3), padding=1, bias=False),
    nn.BatchNorm2d(256),
    ...

![Training and Validation losses and accuracies](/episode03/dermamnist_v6_even_deeper_network/training_val_comparison_v4_v5_v6.png)

In the image above, the blue curves correspond to [v4](/episode03/dermamnist_v4_adam_TB.py), and pink curves are for [v5](/episode03/dermamnist_v5_deeper_network.py), and the green ones are for [the latest version](/episode03/dermamnist_v6_even_deeper_network.py). 

Even though the losses for both training and validation are lower than the previous versions, the overfitting phenomenon still persists (see the validation accuracy curve). This points to the need for regularization: where we make it harder now for the network to memorize, and provide more variations of input data to learn from.

![Test accuracy and classification report](/episode03/dermamnist_v6_even_deeper_network/test_accuracy_v6.png)

For this version, the test accuracy is again lower than the previous one, pointing to the same issues as earlier. In the final version below, we introduce data augmentation to find out if it can resolve these problems.

--------------------

## Seventh version:

Adding data augmentation at the end, to imclude some regularization and possibly improve generalization.

    training_transform_medmnist = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Pad(2),
            transforms.RandomCrop(
                size=(32, 32), padding=(0, 0, 5, 5), padding_mode="reflect"
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]
    )

--------------------

## References

* For more general tips and tricks around model training (general because it isn't Medical Imaging in particular), [Andrei Karpathy's recipe from 2019](https://karpathy.github.io/2019/04/25/recipe/)is highly recommended. The contents of this episode is an extension of this with focus on medical image data sets. 

* [This](https://medium.com/miccai-educational-initiative/project-roadmap-for-the-medical-imaging-student-working-with-deep-learning-351add6066cf) blog post for more great tips while training models. This was a previous entry to the MICCAI Education Challenge as well!

* If you're inclined to use MONAI, consider following [this](https://github.com/Project-MONAI/tutorials/blob/main/2d_classification/mednist_tutorial.ipynb) tutorial. It follows an older version of the medMNIST data set, and uses MONAI to load the data and build models more easily.

* [MMAR](https://docs.nvidia.com/clara/clara-train-sdk/pt/mmar.html) is a Medical Model Archive designed by NVIDIA to organize all artifacts produced during the model development life cycle. This may be too heavyweight for research prototypes that one may begin with, but as things become more stable, such standards (much like BIDS for data storage) may help avoid further pains down the line.

* Consider following the [MICCAI Hackathon reproducibility checklist](https://github.com/JunMa11/MICCAI-Reproducibility-Checklist) to ensure that your pipeline is not too exotic, and future researchers can build on your work!

For questions/suggestions for improvements, please [create an issue](https://github.com/ubern-mia/bender/issues) in the BENDER repository.