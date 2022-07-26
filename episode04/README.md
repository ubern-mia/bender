# Evaluation and Deployment: final points to consider

At this stage, you have curated the data set, built in an organized fashion: a model that surpasses previously published benchmarks, and are now ready to show it to your clinical collaborators and even write a report/paper for review. 

In this episode of the [BENDER Series](https://github.com/ubern-mia/bender), we go over some final checks to make this translation of research-grade models to something that could be tested in clinics better.

--------------------

## External test data set 

Portability across clinical settings is something of special importance in medical image analysis. It is likely that in deployment, out-of-distribution data, subjects from a different geography/ethnicity, and several other variations could make what is otherwise a State-Of-The-Art model fail miserably. Therefore, it is essential to consider verifying apriori using separately held out data (whenever available) from such scenarios.

* Read more about [out-of-distribution detection](https://ai.googleblog.com/2019/12/improving-out-of-distribution-detection.html) here. 

## Multiple vendors (from an imaging hardware perspective) and varied protocols of acquisition of data


## Average metrics versus robustness and reliability across distributions

Data sets used for training (even if they came from a curated publicly available repository like MedMNIST) are inherently biased. What these Deep Learning models are great at doing is learning patterns: not necessarily only the ones we humans associate with the categories we want to discriminate. Simply achieving the highest average metric on the test set is not a sufficient condition for ensuring reliability and robustness across a variety of image inputs: it could simply mean that this SoTA model you've built has only incrementally learnt to perform better on this test set (sometimes called overfitting the test set, especially in a challenge/competitive modeling setting). 

Hence, running some kind of a robustness test: demonstrating behavior for artificially created worst-case scenarios, indicating not just mean metrics but standard deviation across multiple training runs, and considering adversarial robustness through augmentation or another similar method would be desirable (clearly, this is quite a lot of work and may not always be possible, but it would certainly improve the levels of trust on such systems while deploying).

## Imbalanced testing dataset categories/pixel-labels


## Dealing with reviewer #2 and other situations


--------------------

For questions/suggestions for improvements, please [create an issue](https://github.com/ubern-mia/bender/issues) in the BENDER repository.
