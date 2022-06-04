# Glossary

Deep Learning and especially as applied to Medical Imaging seems to have several terms that are thrown around - which are difficult to really track and remember at all times. This page is meant to include a single line (or only slightly longer) description of some of these commonly used terms, so that reading papers and talking to peers is easier than otherwise.

See https://youtu.be/Gbnep6RJinQ?t=1626 for a discussion around this very topic at Stanford AIMI.

## Clinical terms that could confuse an engineer/technical person

### Exogenous and Endogenous contrast

### hyper-intense, hypo-intense

### Lesions versus tumor versus target volume

### Anterior and Posterior, Dorsal and Ventral, Sagittal, Coronal and Axial, Radial and Lateral, ...

### MRI sequences:
This indicates the various modalities - T1, T1c, T2, FLAIR and so on. There could be one or more that are used for segmentation.

### T1c:
The C here stands for Contrast Enhancement. It appears that liquids and other features are more prominently displayed here. Gadolinium or what are other contrasts? 

## Technical terms that could confuse a clinician

### Image properties related:

#### Saliency map

#### Regularization

#### Optimizer (ADAM, Gradient Descent family of, ...)

#### Image Super-resolution:
Reconstructing a higher-resolution image or image sequence from the observed low-resolution image.

#### Data Augmentation:
Augmenting the training data set can be done by applying various transformations that preserves the labels, as in rotations, scalings and intensity shifts of images, or more advanced data augmentation techniques like anatomically sound deformations (for medical image analysis), or other data set specific operations.

### Deep Learning model training related:

#### Model warmup and validation patience

#### Batch Normalization Layer:
These layers are typically placed after activation layers, producing normalized activation maps by subtracting the mean and dividing by the standard deviation for each training batch. Including batch normalization layers forces the network to periodically change its activations to zero mean and unit standard deviation as the training batch hits these layers, which works as a regularizer for the network, speeds up training, and makes it less dependent on careful parameter initialization. 

#### Dropout Layer:
By averaging several models in an ensemble one tend to get better performance than when using single models. Dropout is an averaging technique based on stochastic sampling of neural networks. By randomly removing neurons during training one ends up using slightly different networks for each batch of training data, and the weights of the trained network are tuned based on optimization of multiple variations of the network.

#### Skip connections:
These make it possible to train much deeper networks. A 152 layer deep ResNet won the 2015 ILSVRC competition, and the authors also successfully trained a version with 1001 layers. Having skip connections in addition to the standard pathway gives the network the option to simply copy the activations from layer to layer (more precisely, from ResNet block to ResNet block), preserving information as data goes through the layers.

#### Dense Nets:
These build on the ideas of ResNet, but instead of adding the activations produced by one layer to later layers, they are simply concatenated together. The original inputs in addition to the activations from previous layers are therefore kept at each layer (again, more precisely, between blocks of layers), preserving some kind of global state. This encourages feature reuse and lowers the number of parameters for a given depth. DenseNets are therefore particularly well-suited for smaller data sets (outperforming others on e.g. Cifar-10 and Cifar-100)

#### Generative Adversarial Networks:
A generative adversarial network consists of two neural networks pitted against each other. The generative network G is tasked with creating samples that the discriminative network D is supposed to classify as coming from the generative network or the training data. The networks are trained simultaneously, where G aims to maximize the probability that D makes a mistake while D aims for high classification accuracy.

#### Siamese Networks:
A Siamese network consists of two identical neural networks, both the architecture and the weights, attached at the end. They are trained together to differentiate pairs of inputs. Once trained, the features of the networks can be used to perform one-shot learning without retraining

### Types of learning:

#### Curriculum Learning: 
the model starts learning from easy scenarios before turning to more difficult ones. (several variations around this!)

#### Teacher-student network training:
This has something to do with transfer learning - the teacher is like the pre-trained network, and the student is like the new problem statement + re-training/transfer domain adaptation thing.

#### Bayesian Deep Learning:
One way to increase their trustworthiness is to make them produce robust uncertainty estimates in addition to predictions. The field of Bayesian Deep Learning aims to
combine deep learning and Bayesian approaches to uncertainty. 

#### Transfer Learning: 
Also called fine-tuning or pre-training: first you train a network to perform a task where there is an abundance of data, and then you copy weights from this network to a network designed for the task at hand.

An interesting example is to do interorgan transfer learning in 3D, an idea we have used for kidney segmentation, where pre-training a network to do brain segmentation decreased the number of annotated kidneys needed to achieve good segmentation performance. 

#### Federated Learning (or Differential Privacy): 
Most current work on deep learning for medical data analysis use either open, anonymized data sets, or locally obtained anonymized research data, making these issues less relevant. However, the general deep learning community are focusing a lot of attention on the issue of privacy, and new techniques and frameworks for federated learning and differential privacy are rapidly improving. 
