# Neural Network Framework from Scratch

This repository contains a fully implemented neural network framework built from scratch, including training and evaluation on multiple datasets. The framework supports standard feedforward networks and a variation of ResNet with skip connections. The goal of this project was to understand deep learning frameworks' inner workings, such as backpropagation, optimization techniques, and hyperparameter tuning.

## Features
- Fully connected feedforward neural networks (Standard NN)
- Residual Neural Networks (ResNet)
- Gradient testing for verification
- Softmax regression implementation
- SGD optimizer with momentum
- Hyperparameter tuning and grid search
- Support for multiple datasets (Peaks, GMM, SwissRoll)

## Mathematical Formulation
The neural network is trained by minimizing the softmax loss function:

```math
L(\{\theta_l\}_{l=1}^{L}) = \frac{1}{m} \sum_{i=1}^{m} \ell(\theta^{(L)}, x_i^{(L)}, y_i)
```

Where:

```math
\ell() \text{ is the softmax objective function}
```

```math
x_i^{(1)} \text{ represents the input data}
```

- The network layers are defined recursively as:
### Standard Feedforward Network:
```math
f(\theta^{(l)}, x_i^{(l)}) = \sigma(W^{(l)} x_i^{(l)} + b^{(l)})
```

### Residual Network:
```math
f(\theta^{(l)}, x_i^{(l)}) = x_i^{(l)} + W_2^{(l)} \sigma(W_1^{(l)} x_i^{(l)} + b_1^{(l)}) + b_2^{(l)}
```

## Implementation
The core implementation is divided into the following files:
- `nn.py`: Contains the implementation of the neural network architecture, including forward and backward propagation.
- `main.py`: Runs experiments on different datasets and evaluates model performance.

### Optimizer (SGD with Momentum)
The optimization of the network is done using Stochastic Gradient Descent (SGD) with momentum:

```math
v_t = \beta v_{t-1} + (1-\beta) \nabla_{\theta} J(\theta)
```
```math
\theta = \theta - \alpha v_t
```

where \( \alpha \) is the learning rate and \( \beta \) is the momentum term.

### Gradient Testing
The correctness of the gradients was validated using the first-order and second-order Taylor approximations. The gradient test is implemented under the `GradientTest` class.

## Datasets
This framework has been tested on multiple datasets:
1. **Peaks Data (2D, 5 Classes)**
2. **GMM Data (5D, 5 Classes)**
3. **SwissRoll Data (2D, 2 Classes)**

### Best Performance Summary
- **Peaks Dataset:** Standard NN achieves ~56.5% accuracy, while ResNet outperforms it with higher stability.
- **GMM Dataset:** Easier to classify; both architectures perform well, but depth variations impact performance.

## Hyperparameter Tuning
A grid search was performed with the following parameters:
- Learning rate \( \alpha \in \{1, 0.5, 0.1, 0.01, 0.001\} \)
- Batch sizes \( \in \{1, 10, 50, 100, 500, 1000\} \)
- Number of layers: Varies for Standard NN and ResNet

Best results were obtained with:
- Batch size: 50-100
- Learning rate: 0.5-0.1
- Alpha decay applied every 100 iterations resulted in smoother convergence.

## Usage
To train the network, run the following:
```bash
python main.py --dataset peaks --epochs 1000 --batch_size 100 --lr 0.1
```

For gradient testing:
```bash
python main.py --test_gradients
```

## Results and Observations
1. **Gradient tests confirmed correctness of backpropagation implementation.**
2. **Smaller batch sizes led to unstable training, while larger batches improved stability.**
3. **ResNet consistently outperformed Standard NN due to skip connections.**
4. **Alpha decay improved training efficiency and convergence.**

## Conclusion
This project demonstrates the feasibility of implementing a neural network framework from scratch, including forward and backward propagation, optimization, and dataset classification. The results align with expected deep learning behaviors, highlighting the benefits of architectures like ResNet and hyperparameter tuning.

