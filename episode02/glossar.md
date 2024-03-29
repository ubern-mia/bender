# Glossary

Deep Learning and especially as applied to Medical Imaging seems to have a lot of jargon - which is difficult to track and remember at all times. This page, part of the [BENDER Series](https://github.com/ubern-mia/bender), is meant to include a single line (or only slightly longer) description of some of these commonly used terms, so that reading papers and talking to research collaborators is easier than otherwise.

See [here](https://youtu.be/Gbnep6RJinQ?t=1626) for an interesting discussion around this very topic.

--------------------

# Table of Contents
1. [Clinical Terms for Technical folks](#clinical_terms)
    - [Anatomical directions](#anatomical_directions)
    - [Planes of the body](#planes_of_the_body)
    - [Computed Tomography](#computed_tomography)
    - [Magnetic Resonance Imaging](#MRI)
    - [Functional MRI](#fMRI)
    - [MRI Sequences](#mri_sequences)
    - [Contrast agents](#t1_contrast)
    - [Hyper/Hypo Intense](#hyper_hypo_intense)
2. [Technical Terms for Clinical folks](#technical_terms)
    - [AI](#ai)
    - [Machine Learning](#machine_learning)
    - [Deep Learning](#deep_learning)
    - [Neural Networks](#neural_network)
    - [Convolutional Neural Network](#cnn)
    - [Image Classification](#image_classification)
    - [Image Segmentation](#image_segmentation)
    - [U-Net](#unet)
    - [Dense Net](#dense_net)
    - [Generative Adversarial Networks](#GAN)
    - [Siamese Networks](#siamese_networks)
    - [Transfer Learning](#transfer_learning)
    - [Federated Learning](#federated_learning)
    - [Skip Connections](#skip_connections)
    - [Super-resolution](#superresolution)
    - [Data Augmentation](#data_augmentation)
3. [References](#references)

--------------------

## Clinical terms for Technical Folks <a name="clinical_terms"></a>

### Anatomical directions (from [here](https://training.seer.cancer.gov/anatomy/body/terminology.html)) <a name="anatomical_directions"></a>

Superior (general) or cranial (brain/neuro-focused) - toward the head end of the body; upper (example, the hand is part of the superior extremity).

Inferior (general) or caudal (brain/neuro-focused) - away from the head; lower (example, the foot is part of the inferior extremity).

Anterior or ventral - front (example, the kneecap is located on the anterior side of the leg).

Posterior or dorsal - back (example, the shoulder blades are located on the posterior side of the body).

Medial - toward the midline of the body (example, the middle toe is located at the medial side of the foot).

Lateral - away from the midline of the body (example, the little toe is located at the lateral side of the foot).

Proximal - toward or nearest the trunk or the point of origin of a part (example, the proximal end of the femur joins with the pelvic bone).

Distal - away from or farthest from the trunk or the point or origin of a part (example, the hand is located at the distal end of the forearm).

### Planes of the body (from [here](https://training.seer.cancer.gov/anatomy/body/terminology.html)) <a name="planes_of_the_body"></a>

Coronal Plane (Frontal Plane) - A vertical plane running from side to side; divides the body or any of its parts into anterior and posterior portions.

Sagittal Plane (Lateral Plane) - A vertical plane running from front to back; divides the body or any of its parts into right and left sides.

Median plane (a special case of Sagittal plane) through the midline of the body; divides the body or any of its parts into right and left halves.

Axial Plane (Transverse Plane) - A horizontal plane; divides the body or any of its parts into upper and lower parts.

### Computed Tomography (from [here](https://www.nibib.nih.gov/science-education/glossary)) <a name="computed_tomography"></a>
A computerized X-ray imaging procedure in which a narrow beam of X-rays is aimed at a patient and quickly rotated around the body, producing signals that are processed by the machine’s computer to generate cross-sectional images—or “slices”—of the body. These slices are called tomographic images and contain more detailed information about the internal organs than conventional X-rays. 

### Magnetic Resonance Imaging (from [here](https://www.nibib.nih.gov/science-education/glossary)) <a name="MRI"></a>
A non-invasive imaging technology used to investigate anatomy and function of the body in both health and disease without the use of damaging ionizing radiation. It is often used for disease detection, diagnosis, and treatment monitoring. It is based on sophisticated technology that excites and detects changes in protons found in the water that makes up living tissues.

### fMRI (functional MRI) (from [here](https://www.nibib.nih.gov/science-education/glossary)) <a name="fMRI"></a>
An MRI-based technique for measuring brain activity. It works by detecting the changes in blood oxygenation and flow that occur in response to neural activity – when a brain area is more active it consumes more oxygen and to meet this increased demand blood flow increases to the active area. fMRI can be used to produce activation maps showing which parts of the brain are involved in a particular mental process.

### MRI sequences <a name="mri_sequences"></a>
This indicates the various modalities - T1, T1c, T2, FLAIR and so on. There could be one or more that are used for segmentation. These sequences are generated by varying the repetition and echo times while capturing the MR signals. For more details, see [here](https://mriquestions.com/tr-and-te.html). FLAIR stands for FLuid Attenuated Inversion Recovery.

### T1c <a name="t1_contrast"></a>
The C here stands for Contrast Enhancement. These are scans where specific contrast agents are used (like Gadolinium) to enhance the how certain anatomy shows up in imaging scans, and only act temporarily during the period of the scan. There are several contrast agents, and a more detailed explanation is [here](https://mriquestions.com/hellipcontrast-agents--blood.html) and [here](https://www.radiologyinfo.org/en/info/safety-contrast).

### hyper-intense, hypo-intense <a name="hyper_hypo_intense"></a>
hyper-intense translates to higher than normal/expected brightness/intensity, and hypo-intense is lower than normal/expected. Depending on the sequence of imaging and contrast agents used, different parts of the anatomy may show up as hyper/hypo intense compared to their surrounding tissues.

### Exogenous and Endogenous contrast

### Lesions versus tumor versus target volume

--------------------

## Technical terms for Clinical Folks <a name="technical_terms"></a>

### AI (from [here](https://www.tasq.ai/glossary/artificial-intelligence/)) <a name="ai"></a>
The objective of AI is to make computers/computer programs clever enough to mimic the behavior of our brains. This is intentionally a broad term, and Machine Learning, Deep Learning are merely subsets (specific instances of a type of AI) which have recently shown tremendous success, so much so, that Machine Learning is almost synomymous to AI. 

### Machine Learning (from [here](https://www.nibib.nih.gov/science-education/glossary)) <a name="machine_learning"></a>
Machine Learning is a subset/type of AI, where a computer algorithm (a set of rules and procedures) is developed to analyze and make predictions from data that is fed into the system. Machine learning-based technologies are routinely used every day, such as personalized news feeds and traffic prediction maps.

### Deep Learning (from [here](https://www.nibib.nih.gov/science-education/glossary)) <a name="deep_learning"></a>
A form of machine learning that uses many layers of computation to form what is described as a deep neural network, capable of learning from large amounts of complex, unstructured data. Deep neural networks are responsible for voice-controlled virtual assistants as well as self-driving vehicles, which learn to recognize traffic signs.

### (Artificial) Neural Networks (from [here](https://www.nibib.nih.gov/science-education/glossary)) <a name="neural_network"></a>
A machine learning approach modeled after the brain in which algorithms process signals via interconnected nodes called artificial neurons. Mimicking biological nervous systems, artificial neural networks have been used successfully to recognize and predict patterns of neural signals involved in brain function.

### Convolutional Neural Networks (from [here](https://www.tasq.ai/glossary/convolutional-neural-network-cnn/)) <a name="cnn"></a>
A Convolutional Neural Network (CNN) is a Deep Learning system that can take an input picture, give relevance (learnable weights and biases) to various aspects/objects in the image, and distinguish between them. When compared to other classification algorithms, the amount of pre-processing required by a CNN is significantly less. While filters are hand-engineered in basic approaches, CNN can learn these filters/characteristics with adequate training.

### Image Classification (from [here](https://www.tasq.ai/glossary/classification/)) <a name="image_classification"></a>
Classification is described as the operation of identifying, interpreting, and organizing objects into specified groups. A great example of classification is to categorize emails as “spam” or “non-spam,” as employed by today’s leading email service providers. For medical images, this could mean marking the entire image as "healthy" or "cancerous" - without any more details about what makes it cancerous or not. 

### Image Segmentation <a name="image_segmentation"></a>
Segmentation (more specifically, semantic or pixel-wise segmentation) is the operation of classifying each pixel/component of an image into categories. For example, what part of a pathology slide is part of a cell? What are the boundaries of the brain in a MR image? Specialized AI models/systems exist to automatically estimate these boundaries with fairly high accuracies compared to humans, at a fraction of the time/cost. 

### U-Net (from [here](https://www.tasq.ai/glossary/u-net/)) <a name="unet"></a>
U-net was created for neural network image segmentation and was the first to use it. Its design is roughly divided into two parts: an encoder network and a decoder network. Unlike classification, where the U-net network’s final output is the only thing that matters, semantic segmentation necessitates not just pixel-level discrimination but also a technique to construct the discriminative features learned at various stages of the encoder onto the image pixels.

### Dense Net <a name="dense_net"></a>
These build on the ideas of ResNet, but instead of adding the activations produced by one layer to later layers, they are simply concatenated together. The original inputs in addition to the activations from previous layers are therefore kept at each layer (again, more precisely, between blocks of layers), preserving some kind of global state. This encourages feature reuse and lowers the number of parameters for a given depth. DenseNets are therefore particularly well-suited for smaller data sets (outperforming others on e.g. Cifar-10 and Cifar-100)

### Generative Adversarial Networks <a name="GAN"></a>
A generative adversarial network consists of two neural networks pitted against each other. The generative network G is tasked with creating samples that the discriminative network D is supposed to classify as coming from the generative network or the training data. The networks are trained simultaneously, where G aims to maximize the probability that D makes a mistake while D aims for high classification accuracy.

### Siamese Networks <a name="siamese_networks"></a>
A Siamese network consists of two identical neural networks, both the architecture and the weights, attached at the end. They are trained together to differentiate pairs of inputs. Once trained, the features of the networks can be used to perform one-shot learning without retraining

### Transfer Learning or fine-tuning <a name="transfer_learning"></a>
Also called fine-tuning or pre-training: first you train a network to perform a task where there is an abundance of data, and then you copy weights from this network to a network designed for the task at hand. An interesting example is to do interorgan transfer learning in 3D, an idea we have used for kidney segmentation, where pre-training a network to do brain segmentation decreased the number of annotated kidneys needed to achieve good segmentation performance. 

### Federated Learning (or Differential Privacy) <a name="federated_learning"></a>
Most current work on deep learning for medical data analysis use either open, anonymized data sets, or locally obtained anonymized research data, making these issues less relevant. However, the general deep learning community are focusing a lot of attention on the issue of privacy, and new techniques and frameworks for federated learning and differential privacy are rapidly improving. 

### Skip connections <a name="skip_connections"></a>
These make it possible to train much deeper networks. A 152 layer deep ResNet won the 2015 ILSVRC competition, and the authors also successfully trained a version with 1001 layers. Having skip connections in addition to the standard pathway gives the network the option to simply copy the activations from layer to layer (more precisely, from ResNet block to ResNet block), preserving information as data goes through the layers.

### Image Super-resolution <a name="superresolution"></a>
Reconstructing a higher-resolution image or image sequence from the observed low-resolution image.

### Data Augmentation <a name="data_augmentation"></a>
Augmenting the training data set can be done by applying various transformations that preserves the labels, as in rotations, scalings and intensity shifts of images, or more advanced data augmentation techniques like anatomically sound deformations (for medical image analysis), or other data set specific operations.

TODO

### Model warmup and validation patience

### Saliency map

### Regularization

### Optimizer (ADAM, Gradient Descent family of, ...)

--------------------

## References <a name="references"></a>

See [Radiopaedia](https://radiopaedia.org/search) for a comprehensive list of clinical terms. [Radiology specific terms](https://radiopaedia.org/articles/terms-used-in-radiology) could be specifically more useful. Here are some more links for clinical terms:

* [Harvard Health](https://www.health.harvard.edu/a-through-c)
* [CDC's glossary of Radiation](https://emergency.cdc.gov/radiation/pdf/glossary.pdf)
* [Radacademy's glossary](https://www.asrt.org/radcademy/glossary) and [set of videos](https://www.asrt.org/radcademy/videos)
* [NIBIB's glossary for Biomedical Imaging](https://www.nibib.nih.gov/science-education/glossary)
* [MRI questions](https://mriquestions.com/complete-list-of-questions.html)

Here are some references for a more exhaustive list of technical terms, some of which we have reproduced here:

* [tasq.ai's Glossary of AI terminology](https://www.tasq.ai/glossary/)
* [NIBIB's glossary of Medical Imaging](https://www.nibib.nih.gov/science-education/glossary)
* [AI: a glossary of terms for AI in Medical Imaging](https://link.springer.com/content/pdf/bbm%3A978-3-319-94878-2%2F1.pdf)

--------------------

For questions/suggestions for improvements, please [create an issue](https://github.com/ubern-mia/bender/issues) in the BENDER repository.
