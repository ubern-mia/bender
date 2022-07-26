# Evaluation and Deployment: final points to consider

At this stage, you have curated the data set, built in an organized fashion: a model that surpasses previously published benchmarks, and are now ready to show it to your clinical collaborators and even write a report/paper for review. 

In this episode of the [BENDER Series](https://github.com/ubern-mia/bender), we go over some final checks to make this translation of research-grade models to something that could be tested in clinics better.

--------------------

## External test data set 

Portability across clinical settings is something of special importance in medical image analysis. It is likely that in deployment, out-of-distribution data, subjects from a different geography/ethnicity, and several other variations could make what is otherwise a State-Of-The-Art model fail miserably. Therefore, it is essential to consider verifying apriori using separately held out data (whenever available) from such scenarios.

Read more about [out-of-distribution detection](https://ai.googleblog.com/2019/12/improving-out-of-distribution-detection.html) here on this Google AI blog. 

## Multiple vendors (from an imaging hardware perspective) and varied protocols of acquisition of data

Training on a data set which was acquired from the same imaging hardware or even the same hospital and acquisition protocol setting is typically called single-source bias. It may so happen that the model we build learns these nuances about the specific machine and acquisition protocol better than the actual characteristics of the category/contour/region that we want it to learn. 

Therefore, it is imperative to evaluate models by varying these generative parameters: choose different hardware vendors (for CT, MR, and so on); have a diverse set of acquisition protocols to make the model agnostic to these variations (sometimes this could be simulated as well!); and most importantly, ensure that the test data set is of high enough quality (not just from a label accuracy perspective, but also that of the breadth of coverage of the actual target deployment distribution): this would help more holistically evaluate the impact of such models in clinics where it would otherwise be tested on such diverse distributions of data in any case. 

See [this paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7104701/) for a deeper dive into some of these considerations.

## Average metrics versus robustness and reliability across distributions

Data sets used for training (even if they came from a curated publicly available repository like MedMNIST) are inherently biased. What these Deep Learning models are great at doing is learning patterns: not necessarily only the ones we humans associate with the categories we want to discriminate. Simply achieving the highest average metric on the test set is not a sufficient condition for ensuring reliability and robustness across a variety of image inputs: it could simply mean that this SoTA model you've built has only incrementally learnt to perform better on this test set (sometimes called overfitting the test set, especially in a challenge/competitive modeling setting). 

Hence, running some kind of a robustness test: demonstrating behavior for artificially created worst-case scenarios, indicating not just mean metrics but standard deviation across multiple training runs, and considering adversarial robustness through augmentation or another similar method would be desirable (clearly, this is quite a lot of work and may not always be possible, but it would certainly improve the levels of trust on such systems while deploying).

See [this recent nature article](https://www.nature.com/articles/s41746-022-00592-y#Sec7) that discusses these effects and how to be mindful of them.

## Imbalanced testing dataset categories/pixel-labels

As seen in the [many versions of models we built in episode 3](/episode03/README.md), medical image data is particularly prone to class imbalance. This problem is even worse for semantic segmentation tasks, where the number of pixels representing anomalous regions is much smaller than healthy/normal regions. There are several ways to handle this in the modeling process, however, while reporting results, equal care must be taken to indicate realistic behavior. Like in the DermaMNIST example, if one of the categories is very dominant in number, it is possible that a network that appears to perform well can simply do well in that specific category and completely avoid all the others (`dermatofibroma`, for example). 

We recommend that metrics of evaluation are reported not just on average, but for all the categories individually as well (ideally, the entire confusion chart or the [classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) is preferred). This would additionally help debug potential unexpected behavior in deployment, which would otherwise not have been caught while training such models. 

## Dealing with reviewer #2 and other situations

Finally, when you think you're ready to write a report/paper, there's always a [reviewer # 2](https://twitter.com/GrumpyReviewer2) lurking around. Here are some tips and tricks from the community to help navigate this labyrinth:

* [Microsoft Research guide to writing a great research paper](https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/)
* [How to write a good paper by Bill Freeman, MIT](https://faculty.cc.gatech.edu/~parikh/citizenofcvpr/static/slides/freeman_how_to_write_papers.pdf)
* [Devi Parikh's guide to rebuttals](https://deviparikh.medium.com/how-we-write-rebuttals-dc84742fece1)
* [How NOT to be reviewer #2 yourself](https://link.springer.com/article/10.1007/s40037-021-00670-z)

--------------------

For questions/suggestions for improvements, please [create an issue](https://github.com/ubern-mia/bender/issues) in the BENDER repository.
