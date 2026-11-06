# Handwritten Digit Classification with CNN Augmentation and Wavelet Scattering

**CS5487 Course Project Proposal**

**Group members:** 张宝坤 (72610491), 王昌浩 (72610116), 彭迪 (72610480)

## Problem and Objective

Handwritten digit recognition is challenging because writers differ in stroke thickness, position, and shape. We will compare three approaches: a pixel-based 1-nearest-neighbor (1-NN) baseline, an augmented convolutional neural network (CNN), and wavelet scattering features with a linear support vector machine (SVM). Our main question is whether learned or fixed image representations generalize better with limited training data, and whether morphological augmentation improves CNN robustness to handwriting variation.

## Dataset and Preprocessing

We will use only the supplied MNIST subset: 4,000 grayscale images of size 28 x 28, with 400 examples per digit. Pixel values will be scaled to [0, 1]. We will follow both prescribed trials, each containing 2,000 training and 2,000 test images, with no shared writers between training and testing within a trial. The instructor's additional handwritten digits will be reserved for the bonus challenge when released.

## Proposed Methods

**1-NN baseline.** We will classify flattened pixel vectors using Euclidean distance and reproduce the provided baseline (91.60% mean accuracy).

**Augmented CNN.** We will train a compact CNN with two convolutional blocks, ReLU activations, pooling, dropout, and a ten-class output. Mild random translation and rotation will provide geometric augmentation. We will additionally test randomly applying either grayscale erosion or dilation to simulate thinner or thicker strokes [1]. Small structuring elements and conservative application probabilities will be chosen using training-only validation and visual checks of training images. Some images will remain unchanged, and transformations must preserve digit identity. One controlled ablation will compare the same CNN with and without morphological augmentation, holding other settings fixed.

**Wavelet scattering + linear SVM.** Using Kymatio [3], we will extract second-order 2-D scattering coefficients with fixed wavelet filters, then standardize the flattened features and train a linear SVM. Scattering offers a representation designed for translation tolerance and stability to small deformations [2], providing a useful comparison with the CNN's learned filters.

## Evaluation and Expected Outcome

Within each trial, a fixed stratified 80/20 split of the prescribed training set will guide limited hyperparameter selection and CNN early stopping. Augmentation will be applied only to training examples; feature scaling will be fitted only on the corresponding training partition. All settings will be frozen before final training and test evaluation. We will report accuracy for each trial and their mean, compare against 1-NN, and inspect confusion matrices and representative errors. A successful outcome is an improvement over the baseline together with evidence about the benefit or harm of morphological augmentation; improvement is not assumed. For the bonus challenge, we will apply the validation-selected pipeline under the instructor's rules without tuning on challenge labels. Implementation will use Python, PyTorch, scikit-learn, and Kymatio.

## References

[1] OpenCV. *Morphological Transformations*. https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

[2] J. Bruna and S. Mallat. *Invariant Scattering Convolution Networks*. arXiv:1203.1513, 2012. https://arxiv.org/abs/1203.1513

[3] M. Andreux et al. *Kymatio: Scattering Transforms in Python*. JMLR, 21(60):1-6, 2020. https://jmlr.org/papers/v21/19-047.html
