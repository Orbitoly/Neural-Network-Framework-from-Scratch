# Neural-Network-Framework-from-Scratch
BSc project - in-depth exploration and implementation of neural networks from the ground up.

## Project Overview

In this project, we present an in-depth exploration and implementation of neural networks from the ground up. Our endeavor is not merely academic—it's a proof of proficiency in the core technologies underlying modern AI. By developing the softmax regression loss function, applying Stochastic Gradient Descent (SGD) with momentum, and experimenting with both standard and Residual Neural Network (ResNet) architectures, we seek to demystify the internal mechanisms of deep learning frameworks, illustrate the nuances of back-propagation, and evaluate the influence of hyperparameter tuning on model performance.

## Contents

- [Neural Network Framework from Scratch](#neural-network-framework-from-scratch-an-in-depth-machine-learning-initiative)
  - [Project Overview](#project-overview)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Implementation Overview](#implementation-overview)
  - [Mathematical Foundations](#mathematical-foundations)
    - [Logistic Regression and Softmax Function](#logistic-regression-and-softmax-function)
    - [Neural Network Architecture](#neural-network-architecture)
    - [Backpropagation and Gradient Descent](#backpropagation-and-gradient-descent)
    - [Loss Functions](#loss-functions)
  - [Technical Breakdown](#technical-breakdown)
    - [Gradient Verification](#gradient-verification)
    - [SGD Optimization](#sgd-optimization)
    - [Softmax Objective Minimization](#softmax-objective-minimization)
    - [Architectural Insights](#architectural-insights)
  - [Experimental Insights](#experimental-insights)
    - [Peaks Data Classification](#peaks-data-classification)
    - [Gaussian Mixture Model (GMM) Challenge](#gaussian-mixture-model-gmm-challenge)
  - [Conclusion](#conclusion)
  - [Execution Guide](#execution-guide)

## Introduction

Unlocking the intricacies within deep learning frameworks can significantly bolster a model's development and troubleshooting capabilities. As part of a scientific computing course, this project targets the foundational pillars of neural network theory to create a classification model that learns to distinguish between small vector groups given distinct datasets.

## Implementation Overview

The codebase is designed to be modular and transparent, showcasing the neural network's development with comprehensive gradient tests and optimization strategies.

- `nn.py`: The core script encapsulating the neural network logic, gradient verification, and optimization algorithms.
- `/reports`: Detailed documentation of each project phase, providing a narrative of testing methods and outcomes.
- `/data`: The datasets leveraged during model training and evaluation stages.

## Mathematical Foundations

This neural network project is rooted deeply in mathematical concepts, which form the bedrock of our algorithmic implementation. The following is a summary of the key mathematical principles applied throughout the development of our neural network model.

### Logistic Regression and Softmax Function

The project starts with logistic regression and extends to multinomial logistic regression or the softmax function for multi-class classification:

- **Logistic Regression**:

  $$
  \sigma(\mathbf{x}, \mathbf{w}) = \frac{1}{1 + e^{-\mathbf{x}^T \mathbf{w}}}
   $$

  The logistic function is employed to model the probability of binary outcomes, which in our network's final layer extends to the softmax function for multi-class scenarios.

- **Softmax Function**:

  $$
  P(y = j | \mathbf{x}) = \frac{e^{\mathbf{x}^T \mathbf{w}_j}}{\sum_{k=1}^{K} e^{\mathbf{x}^T \mathbf{w}_k}}
  $$

  For classification tasks, the softmax function computes the probabilities of each class over all possible classes.

### Neural Network Architecture

The architecture of a neural network can be expressed through a series of transformations of input data `x` using learned weights `θ`:

- **Layer-wise Transformation**:

  $$ \mathbf{x}^{(l+1)} = f_l(\theta^{(l)}, \mathbf{x}^{(l)}) $$

  Each layer `l` applies a function `f` to the input data `x`, parameterized by weights `θ`.

- **Activation Functions**:

  - **ReLU**: \( \text{ReLU}(x) = \max(0, x) \) with its gradient \( \text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases} \)

  - **Softmax**: As previously defined for multi-class output.

### Backpropagation and Gradient Descent

The backpropagation algorithm is vital for training neural networks, which involves computing gradients of the loss function with respect to the weights:

- **Gradient Computation**:

  $$ \frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial y^{(L)}} \frac{\partial y^{(L)}}{\partial y^{(l)}} \frac{\partial y^{(l)}}{\partial W^{(l)}} $$

  The above chain rule allows the computation of gradients layer by layer, from the output back to the inputs.

- **SGD Optimization**:

  In the context of SGD, the weight update rule is mathematically described by:

  $$ W_{t+1} = W_t - \eta \nabla L(W_t) $$

  where `η` is the learning rate and `∇L(W_t)` is the gradient of the loss function with respect to the weights at step `t`.

### Loss Functions

- **Cross-Entropy Loss**: For multi-class classification, we implement the cross-entropy loss, which for the softmax output is defined as:

  $$ L(W) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{l} t_{ij} \log(p_{ij}) $$

  where `m` is the number of samples, `l` is the number of labels, `t` is the target label, and `p` is the predicted probability.

These mathematical principles are meticulously encoded into our neural network's implementation, enabling it to learn from data and make accurate predictions.


## Technical Breakdown

### Gradient Verification

We established the accuracy of our gradient calculations—a cornerstone for any optimization process—via gradient tests for both the softmax regression loss function and the network's parameters.

### SGD Optimization

Our custom-built SGD optimizer, enhanced with momentum, showcased its capability in efficiently locating the objective function's minima, validated through the resolution of a least squares problem.

### Softmax Objective Minimization

The softmax function minimization was achieved using our SGD optimizer on multi-class classification tasks like "Peaks Data" and "GMM". Extensive hyperparameter tuning via grid search enabled the fine-tuning of model performance.

### Architectural Insights

We implemented and compared two neural network designs:
- **Standard Neural Network**: A traditional structure with fully connected layers followed by activation functions, validated through forward and backward propagation.
- **Residual Neural Network (ResNet)**: By integrating skip connections, we enabled the construction of deeper architectures, sidestepping the vanishing gradient dilemma and boosting the model's performance.

## Experimental Insights

### Peaks Data Classification

The network architectures were first evaluated on the Peaks data problem:
- **Standard NN**: A testing accuracy of approximately 56.5% was realized, laying the foundation for model evaluation.
- **ResNet**: Demonstrated superior performance over the standard architecture, thanks to the integration of skip connections that enhanced learning capability.

### Gaussian Mixture Model (GMM) Challenge

The GMM data problem provided a more complex scenario for evaluating our network's learning capacity. Different configurations and hyperparameters were experimented with, leading to insights on the network's adaptability and learning efficiency. The results clearly demonstrate the network's capability to learn, achieving impressive accuracy levels as visualized in the attached graphs.

![Learning Progression on GMM Data](/path/to/learning_progression_gmm.png)

*The graphs elucidate the decline in loss and the corresponding increase in accuracy, indicating successful learning.*

## Conclusion

This deep dive into neural network development underscores the criticality of foundational knowledge in optimizing AI algorithms and network architectures. The tangible learning demonstrated by our model is a testament to the effective application of theoretical concepts in a practical setting—making this project a significant portfolio piece for aspiring AI professionals.

## Execution Guide

Prerequisites include Python 3.x. Clone the repository, and navigate to the project directory:

```bash
python nn.py
```

---




