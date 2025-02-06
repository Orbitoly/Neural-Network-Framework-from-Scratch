# Neural Network Framework from Scratch

## Introduction
This repository contains a neural network framework implemented from scratch, focusing on efficient computation using matrix operations without explicit loops. The key objective is to understand and verify the neural network architecture through gradient and Jacobian tests, ensuring mathematical correctness in forward and backward propagation.

## Key Features
- **Matrix-Based Computation:** Avoids explicit loops for efficiency.
- **Gradient and Jacobian Verification:** Ensures the correctness of the backpropagation.
- **Support for Standard and ResNet Architectures:** Implements both fully connected and residual networks.
- **SGD Optimization with Momentum:** Provides efficient optimization using Stochastic Gradient Descent.
- **Automatic Differentiation Verification:** Uses gradient testing to validate computations.
- **Integrated Testing:** The framework verifies different components together to ensure correctness in full training runs.

## Installation
Clone the repository and ensure you have NumPy and Matplotlib installed:

```sh
pip install numpy matplotlib
```

## Neural Network Architecture
### Forward Propagation
Each layer in the network transforms the input using the function:

\[
x^{(l+1)} = \sigma(W^{(l)} x^{(l)} + b^{(l)})
\]

For ResNet, we modify the transformation with a skip connection:

\[
x^{(l+1)} = x^{(l)} + W_2^{(l)} \sigma(W_1^{(l)} x^{(l)} + b_1^{(l)}) + b_2^{(l)}
\]

where:
- \( W^{(l)} \) are the weight matrices
- \( b^{(l)} \) are the biases
- \( \sigma \) is the activation function

### Backward Propagation
The network updates parameters using the gradient of the loss function:

\[
\nabla W^{(l)} = \frac{\partial L}{\partial W^{(l)}}
\]

This is efficiently computed using matrix operations rather than explicit loops.

## NeuralNetwork Python Class
The core implementation is encapsulated in the `NeuralNetwork` class:

```python

class ActivationFunction:
    @staticmethod
    def tanh(x, derivative=False):
        if derivative:
            return 1 - np.tanh(x) ** 2
        return np.tanh(x)

    @staticmethod
    def relu(x, derivative=False):
        if derivative:
            return np.where(x > 0, 1, 0)
        return np.where(x > 0, x, 0)

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)


class ForwardActivationFunction:
    @staticmethod
    def get_forward_activation(activation, ResNet=False):
        def f(w, x, b=0):
            return activation(w @ x + b)

        def f_res(w1, x, w2, b1=0):
            return x + w2 @ activation(w1 @ x + b1)

        if ResNet:
            return f_res
        else:
            return f

class NeuralNetwork:
    def __init__(self, input_size, layers_dim_size_arr, output_size, activation, ResNet=False,resnet_size=16):
        self.isResNet = ResNet
        self.layers_W = []
        self.layers_b = []
        if activation == "tanh":
            self.activation = ActivationFunction.tanh
        elif activation == "relu":
            self.activation = ActivationFunction.relu
        else:
            raise Exception("activation: " + activation + " is not supported")
        self.forward_func = ForwardActivationFunction.get_forward_activation(self.activation, self.isResNet)
        self.num_of_classes = output_size

        if self.isResNet:
            self.layers_W2 = []

            self.layers_W.append(np.random.randn(resnet_size, input_size))
            self.layers_b.append(np.zeros((resnet_size, 1)))
            self.layers_W2.append(np.zeros((input_size, resnet_size)))
            for i in range(len(layers_dim_size_arr)):

                self.layers_W.append(np.random.randn(layers_dim_size_arr[i], resnet_size))
                self.layers_W2.append(np.random.randn(resnet_size, layers_dim_size_arr[i]))
                self.layers_b.append(np.zeros((layers_dim_size_arr[i], 1)))
                self.layers_W[i] = self.layers_W[i] / layers_dim_size_arr[i]
                self.layers_W2[i] = self.layers_W2[i] / layers_dim_size_arr[i]
                self.layers_b[i] = self.layers_b[i] / layers_dim_size_arr[i]

            # Linear Separator Layer
            self.layers_W.append(np.random.randn(output_size, resnet_size))
            self.layers_b.append(np.zeros((output_size, 1)))
            self.layers_W[-1] = self.layers_W[-1] / output_size

        else:
            for i in range(len(layers_dim_size_arr)):
                if i == 0:
                    self.layers_W.append(np.random.randn(layers_dim_size_arr[0], input_size))
                    self.layers_b.append(np.zeros((layers_dim_size_arr[0], 1)))
                else:
                    self.layers_W.append(np.random.randn(layers_dim_size_arr[i], layers_dim_size_arr[i - 1]))
                    self.layers_b.append(np.zeros((layers_dim_size_arr[i], 1)))

                self.layers_W[i] = self.layers_W[i] / layers_dim_size_arr[i]
                self.layers_b[i] = self.layers_b[i] / layers_dim_size_arr[i]

            # Linear Separator Layer
            self.layers_W.append(np.random.randn(output_size, layers_dim_size_arr[-1]))
            self.layers_b.append(np.zeros((output_size, 1)))
            self.layers_W[-1] = self.layers_W[-1] / output_size

        self.x_layers = []

    def forward(self, x):
        last_layer_i = len(self.layers_W) - 1
        self.x_layers = [x]

        for i in range(len(self.layers_W)):
            if i != last_layer_i:
                if self.isResNet:
                    if i == 0:
                        changeDimForward = ForwardActivationFunction.get_forward_activation(self.activation, False)
                        x_next = changeDimForward(self.layers_W[i], self.x_layers[i], self.layers_b[i])
                    else:
                        x_next = self.forward_func(self.layers_W[i], self.x_layers[i], self.layers_W2[i], self.layers_b[i])
                else:
                    x_next = self.forward_func(self.layers_W[i], self.x_layers[i], self.layers_b[i])
            else:
                x_next = self.layers_W[i] @ self.x_layers[i] + self.layers_b[i]
                x_next = ActivationFunction.softmax(x_next)

            self.x_layers.append(x_next)

    def backward(self, y) -> np.array:
        gradient = []
        gradient_b = []
        v = None
        last_layer_i = len(self.layers_W) - 1

        if self.isResNet:
            gradient_W2 = []
            for i in range(last_layer_i, -1, -1):
                if i == last_layer_i:
                    gradient_loss_W = gradient_W(self.x_layers[i], self.layers_W[i].T, y)
                    v = gradient_X(self.x_layers[i], self.layers_W[i].T, y)
                    gradient.append(gradient_loss_W.T)
                    gradient_b.append(np.zeros((self.layers_b[i].shape)))
                elif i == 0:
                    jacobian_fi_theta, jacobian_fi_x, jacobian_fi_b = get_Jacobian_Fi(self.layers_W[i],
                                                                                      self.x_layers[i], v,
                                                                                      self.activation, False,
                                                                                      None, self.layers_b[i])
                    v = jacobian_fi_x
                    gradient.append(jacobian_fi_theta)
                    gradient_b.append(jacobian_fi_b)
                    gradient_W2.append(np.zeros(self.layers_W2[i].shape))
                else:
                    jacobian_fi_theta, jacobian_fi_x, jacobian_fi_b = get_Jacobian_Fi(self.layers_W[i],
                                                                                      self.x_layers[i], v,
                                                                                      self.activation, self.isResNet,
                                                                                      self.layers_W2[i],
                                                                                      self.layers_b[i])
                    v = jacobian_fi_x
                    gradient.append(jacobian_fi_theta[0])
                    gradient_W2.append(jacobian_fi_theta[1])
                    gradient_b.append(jacobian_fi_b)
            gradient = gradient[::-1]
            gradient_W2 = gradient_W2[::-1]
            gradient_b = gradient_b[::-1]
            return gradient, gradient_W2, gradient_b

        else:
            for i in range(last_layer_i, -1, -1):
                if i == last_layer_i:
                    gradient_loss_W = gradient_W(self.x_layers[i], self.layers_W[i].T, y)
                    v = gradient_X(self.x_layers[i], self.layers_W[i].T, y)
                    gradient.append(gradient_loss_W.T)
                    gradient_b.append(np.zeros((self.layers_b[i].shape[0], 1)))


                else:
                    jacobian_fi_theta, jacobian_fi_x, jacobian_fi_b = get_Jacobian_Fi(self.layers_W[i],
                                                                                      self.x_layers[i], v,
                                                                                      self.activation, self.isResNet,
                                                                                      None, self.layers_b[i])
                    v = jacobian_fi_x
                    gradient.append(jacobian_fi_theta)
                    gradient_b.append(jacobian_fi_b)

            gradient = gradient[::-1]
            gradient_b = gradient_b[::-1]
            return gradient, gradient_b

   
    


```

### Training the Network
```python
 def fit(self, X, y, alpha, iterations, delta, batch_size):
        if X.shape[1] != y.shape[0]:
            raise Exception("X size: " + str(X.shape[1]) + " is not equal to y size: " + str(y.shape[0]))

        if len(np.unique(y)) > self.num_of_classes:
            raise Exception(
                "num of unique values in y: " + str(len(np.unique(y))) + " is greater than num of classes: " + str(
                    self.num_of_classes))

        beta = 0.9
        v_W = []
        v_b = []
        v_W2 = []
        for i in range(len(self.layers_W)):
            v_W.append(np.zeros(self.layers_W[i].shape))
            v_b.append(np.zeros(self.layers_b[i].shape))

        if self.isResNet:
            for i in range(len(self.layers_W2)):
                v_W2.append(np.zeros(self.layers_W2[i].shape))

        num_epochs = iterations
        indexes = np.arange(X.shape[1])
        for epoch in range(num_epochs):
            np.random.shuffle(indexes)
            batches = np.array_split(indexes, len(indexes) // batch_size)
            for batch in batches:
                X_batch = X[:, batch]
                y_batch = y[batch]
                self.forward(X_batch)

                if self.isResNet:
                    gradi_W1, gradi_W2, gradi_b = self.backward(y_batch)  # [gradiW1, gradiW2]

                    for i in range(len(gradi_W1)):
                        gradi_W1[i] = gradi_W1[i] / len(X_batch)
                    for i in range(len(gradi_W2)):
                        gradi_W2[i] = gradi_W2[i] / len(X_batch)

                    for i in range(len(gradi_b)):
                        gradi_b[i] = gradi_b[i] / len(X_batch)

                    for j in range(len(self.layers_W)):
                        v_W[j] = beta * v_W[j] + (1 - beta) * gradi_W1[j]
                        self.layers_W[j] -= alpha * v_W[j]
                        # self.layers_W[j] -= alpha * gradi_W1[j]

                    for j in range(len(self.layers_W2)):
                        v_W2[j] = beta * v_W2[j] + (1 - beta) * gradi_W2[j]
                        self.layers_W2[j] -= alpha * v_W2[j]
                        # self.layers_W2[j] -= alpha * gradi_W2[j]

                    for j in range(len(self.layers_b)):
                        v_b[j] = beta * v_b[j] + (1 - beta) * gradi_b[j].reshape(-1, 1)
                        self.layers_b[j] -= alpha * v_b[j]
                        # self.layers_b[j] -= alpha * gradi_b[j].reshape(-1, 1)
                else:
                    gradi, gradi_b = self.backward(y_batch)  # / len(X_batch)
                    for i in range(len(gradi)):
                        gradi[i] = gradi[i] / len(X_batch)

                    for i in range(len(gradi_b)):
                        gradi_b[i] = gradi_b[i] / len(X_batch)

                    for j in range(len(self.layers_W)):
                        #using v for momentum:
                        v_W[j] = beta * v_W[j] + (1 - beta) * gradi[j]
                        self.layers_W[j] -= alpha * v_W[j]
                        # self.layers_W[j] -= alpha * gradi[j]

                    for j in range(len(self.layers_b)):
                        #using v for momentum:
                        v_b[j] = beta * v_b[j] + (1 - beta) * gradi_b[j].reshape(-1, 1)
                        self.layers_b[j] -= alpha * v_b[j]
                        # self.layers_b[j] -= alpha * gradi_b[j].reshape(-1, 1)



```

### Making Predictions
```python
    def predict(self, X):
        self.forward(X)
        predict_Y = np.argmax(self.x_layers[-1], axis=0)
        return predict_Y
```

## Use Case Example
Here is an example of defining and training a neural network with specific layer sizes:

```python
input_features=20
output_classes=3
nn = NeuralNetwork(input_features, [5,10,5], output_classes, "relu", ResNet=useResNet)
print(nn) # 20->10->5->3

nn.fit(data, targets, epochs=1000, learning_rate=0.01)
output = nn.predict(data)
```

This configuration defines a neural network with:
- **20 input neurons**
- **One hidden layer with 5 neurons**
- **One hidden layer with 10 neurons**
- **One hidden layer with 5 neurons**
- **3 output neurons**

## Performance Analysis
Experiments were conducted with different batch sizes and learning rates. The best performance was achieved with:

- **Batch Size:** {50, 100}
- **Learning Rate:** {0.5, 0.1}
- **Gradient Verification:** Passed with errors around \(10^{-6}\)
- **Full Network Testing:** Combined tests confirm end-to-end correctness

## Conclusion
This project provides an efficient and mathematically verified implementation of neural networks from scratch, demonstrating key concepts like forward propagation, backpropagation, gradient testing, and optimization techniques.

## References
1. Scientific Computing Course - Ben-Gurion University
2. Deep Learning by Ian Goodfellow


## Datasets
This framework has been tested on multiple datasets:
1. **Peaks Data (2D, 5 Classes)**
2. **GMM Data (5D, 5 Classes)**
3. **SwissRoll Data (2D, 2 Classes)**

### Best Performance Summary
<img width="827" alt="image" src="https://github.com/user-attachments/assets/bd1cfc14-52ec-4748-b436-3c1c33fcc9de" />
<img width="668" alt="image" src="https://github.com/user-attachments/assets/b3173778-584f-4b6f-a6b8-73607dd3b223" />

tandard NN due to skip connections.**
4. **Alpha decay improved training efficiency and convergence.**


