import numpy.random as random
import numpy as np
from matplotlib import pyplot as plt
import scipy.io
import copy
from main import F, getCMatrix, gradient_W, gradient_X


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

    def predict(self, X):
        self.forward(X)
        predict_Y = np.argmax(self.x_layers[-1], axis=0)
        return predict_Y

    def __str__(self):
        out_str = "W: "
        out_str_w2 = ""
        if self.isResNet:
            out_str = "W1: "
            out_str_w2 += "W2: "
            for i in range(len(self.layers_W)-1):
                out_str += f"{self.layers_W[i].T.shape}"
                out_str_w2 += f"{self.layers_W2[i].T.shape}"
                if i != len(self.layers_W) - 2:
                    out_str += "→"
                    out_str_w2 += "→"
                if i == 0:
                    out_str_w2 = "W2:" + out_str_w2.replace(out_str_w2, ' '*(len(out_str_w2)+3))

            out_str += "→" + f"{self.layers_W[-1].T.shape}"

        else:
            for i in range(len(self.layers_W)):
                out_str += f"{self.layers_W[i].T.shape}"
                if i != len(self.layers_W)-1:
                    out_str += "→"

        out_str += "\n"
        out_str += out_str_w2

        return out_str


def sgd_momentum(X, y, W, gradi, alpha, iterations, delta, beta, batch_size):
    # TODO: use delta to stop
    W_iters = []
    v = np.zeros_like(W)
    num_epochs = iterations
    indexes = np.arange(X.shape[1])

    for epoch in range(num_epochs):
        print("Epoch:", epoch + 1)
        np.random.shuffle(indexes)
        batches = np.array_split(indexes, len(indexes) // batch_size)
        for batch in batches:
            X_batch = X[:, batch]
            y_batch = y[batch]

            print("SGD Momentum:")
            print("X_batch.shape:", X_batch.shape)
            print("y_batch.shape:", y_batch.shape)
            print("W.shape:", W.shape)
            g = gradi(X_batch, y_batch, W)
            g /= batch_size

            v = beta * v + (1 - beta) * g
            W -= alpha * v

        W_iters.append(W.copy())

    return W_iters[-1], W_iters


def get_Jacobian_Fi(W, x, v, activation, isResNet=False, W2=None, b=None):
    if isResNet:
        jacobian_fi_W1 = get_Jacobian_W(activation, ResNet=True)[0](W, x, v, W2, b)
        jacobian_fi_W2 = get_Jacobian_W(activation, ResNet=True)[1](W, x, v, b)
        jacobian_fi_x = get_Jacobian_X(activation, ResNet=True)(W, x, v, W2)
        jacobian_fi_b = get_Jacobian_b(activation, ResNet=True)(W, x, v, W2, b)
        return (jacobian_fi_W1, jacobian_fi_W2), jacobian_fi_x, jacobian_fi_b
    else:
        jacobian_fi_W = get_Jacobian_W(activation, isResNet)(W, x, v, b)
        jacobian_fi_x = get_Jacobian_X(activation, isResNet)(W, x, v, b)
        jacobian_fi_b = get_Jacobian_b(activation, isResNet)(W, x, v, b)
        return jacobian_fi_W, jacobian_fi_x, jacobian_fi_b


def get_Jacobian_W(activation, ResNet=False):
    def activation_prime(x):
        return activation(x, derivative=True)

    def Jacobian_W(w, x, v, b):
        return (activation_prime(w @ x + b) * v) @ x.T

    def Jacobian_W1_Resnet(w1, x, v, w2, b):
        return (activation_prime(w1 @ x + b) * (w2.T @ v)) @ x.T

    def Jacobian_W2_Resnet(w1, x, v, b):
        return v @ activation(w1 @ x + b).T

    if ResNet:
        return Jacobian_W1_Resnet, Jacobian_W2_Resnet
    else:
        return Jacobian_W


def get_Jacobian_X(activation, ResNet=False):
    def activation_prime(x):
        return activation(x, derivative=True)

    def Jacobian_X(w, x, v, b):
        return w.T @ (activation_prime(w @ x + b) * v)

    def Jacobian_X_Resnet(w1, x, v, w2):
        return v + w1.T @ (activation_prime(w1 @ x) * (w2.T @ v))

    if ResNet:
        return Jacobian_X_Resnet
    else:
        return Jacobian_X


def get_Jacobian_b(activation, ResNet=False):
    def activation_prime(x):
        return activation(x, derivative=True)

    if ResNet:
        def Jacobian_b(w1, x, v, w2, b):
            return (activation_prime(w1 @ x + b) * (w2.T @ v)).sum(axis=1)
        return Jacobian_b
    else:
        def Jacobian_b(w, x, v, b):
            return (activation_prime(w @ x + b) * v).sum(axis=1)

        return Jacobian_b


class GradientTest:
    #region Standard NN
    @staticmethod
    def gradient_test_W(Func, jacobian):
        """
        successful gradient test is when the first order is steeper than the zero order
        """
        n = 20
        k = 10
        samples = 10
        x = random.rand(n, samples)
        w = random.rand(k, n)  # (k x n)
        d = random.rand(k, n)  # size of w
        u = random.rand(k, samples)
        b = random.rand(k, 1)
        # y = random.randint(2, size=n) # size of v
        epsilon = 0.1
        F0 = Func(w, x, u, b)
        g0 = jacobian(w, x, u, b)

        iterations = 8
        y0 = np.zeros(iterations)
        y1 = np.zeros(iterations)

        for k in range(iterations):
            epsk = epsilon * pow(0.5, k)
            Fk = Func(w + epsk * d, x, u, b)
            F1 = F0 + epsk * np.dot(g0.flatten(), d.flatten())
            y0[k] = np.abs(Fk - F0)
            y1[k] = np.abs(Fk - F1)

        plt.semilogy(y0)
        plt.semilogy(y1)

        plt.xlabel("Iterations")
        plt.ylabel("error")
        plt.title("Jacobian Transposed Test - W - Standard NN")
        plt.legend(["zero order", "first order"])  # first order should be is steeper
        plt.show()

    @staticmethod
    def gradient_test_X(Func, jacobian):
        """
        successful gradient test is when the first order is steeper than the zero order
        """
        n = 20
        k = 10
        samples = 10
        x = random.rand(n, samples)
        w = random.rand(k, n)  # (k x n)
        d = random.rand(n, samples)  # size of x
        u = random.rand(k, samples)
        b = random.rand(k, 1)
        # y = random.randint(2, size=n) # size of v
        epsilon = 0.1
        F0 = Func(w, x, u, b)
        g0 = jacobian(w, x, u, b)
        iterations = 8
        y0 = np.zeros(iterations)
        y1 = np.zeros(iterations)

        for k in range(iterations):
            epsk = epsilon * pow(0.5, k)
            Fk = Func(w, x + epsk * d, u, b)
            F1 = F0 + epsk * np.dot(g0.flatten(), d.flatten())
            y0[k] = np.abs(Fk - F0)
            y1[k] = np.abs(Fk - F1)

        plt.semilogy(y0)
        plt.semilogy(y1)

        plt.xlabel("Iterations")
        plt.ylabel("error")
        plt.title("Jacobian Transposed Test - X - Standard NN")
        plt.legend(["zero order", "first order"])  # first order should be is steeper
        plt.show()

    @staticmethod
    def gradient_test_b(Func, jacobian):
        """
        successful gradient test is when the first order is steeper than the zero order
        """
        n = 20
        k = 10
        samples = 7
        x = random.rand(n, samples)
        w = random.rand(k, n)  # (k x n)
        d = random.rand(k, 1)  # size of x
        u = random.rand(k, samples)
        b = random.rand(k, 1)
        # y = random.randint(2, size=n) # size of v
        epsilon = 0.1
        F0 = Func(w, x, u, b)
        g0 = jacobian(w, x, u, b)
        iterations = 8
        y0 = np.zeros(iterations)
        y1 = np.zeros(iterations)

        for k in range(iterations):
            epsk = epsilon * pow(0.5, k)
            Fk = Func(w, x, u, b + epsk * d)
            F1 = F0 + epsk * np.dot(g0.flatten(), d.flatten())
            y0[k] = np.abs(Fk - F0)
            y1[k] = np.abs(Fk - F1)

        plt.semilogy(y0)
        plt.semilogy(y1)

        plt.xlabel("Iterations")
        plt.ylabel("error")
        plt.title("Jacobian Transposed Test - b - Standard NN")
        plt.legend(["zero order", "first order"])  # first order should be is steeper
        plt.show()

    @staticmethod
    def direct_jacobian_transposed_test_NN():
        """
        successful jacobian test is when the first order is steeper than the zero order
        """
        activation = ActivationFunction.tanh
        f = ForwardActivationFunction.get_forward_activation(activation)  # forward

        def g(w, x, v, b):
            return np.inner(np.squeeze(f(w, x, b)).flatten(), np.squeeze(v).flatten())

        Jaco_W = get_Jacobian_W(activation)
        Jaco_X = get_Jacobian_X(activation)
        Jaco_b = get_Jacobian_b(activation)

        GradientTest.gradient_test_W(g, Jaco_W)
        GradientTest.gradient_test_X(g, Jaco_X)
        GradientTest.gradient_test_b(g, Jaco_b)

    @staticmethod
    def gradient_test_FC_nn():
        """
        """
        classes = 5
        input_features = 4
        samples = 10
        epsilon = 0.1

        nn = NeuralNetwork(input_features, [5, 3], classes, "tanh")

        x = random.randn(input_features, samples)
        y = random.randint(classes, size=samples)
        C = getCMatrix(y, nn.num_of_classes).T
        d = [np.random.randn(5, input_features), np.random.randn(3, 5), np.random.randn(classes, 3)]
        # db = [np.random.randn(5, 1), np.random.randn(3, 1), np.random.randn(classes, 1)]

        # db = [np.zeros((5, 1)), np.zeros((3, 1)), np.zeros((classes, 1))]
        db = [np.random.randn(5, 1), np.random.randn(3, 1), np.random.randn(classes, 1)]

        nn.forward(x)

        F0 = F(nn.x_layers[-2], nn.layers_W[-1].T, C)
        g0, g0_b = nn.backward(y)

        iterations = 20
        y0 = np.zeros(iterations)
        y1 = np.zeros(iterations)

        for k in range(iterations):
            epsk = epsilon * pow(0.5, k)
            copy_layers_W = copy.deepcopy(nn.layers_W)
            copy_layers_b = copy.deepcopy(nn.layers_b)
            for i in range(len(d)):
                nn.layers_W[i] += epsk * d[i]

            for i in range(len(db)):
                nn.layers_b[i] += epsk * db[i]

            nn.forward(x)

            Fk = F(nn.x_layers[-2], nn.layers_W[-1].T, C)
            gradi_product = 0

            for i in range(len(g0)):
                gradi_product += np.dot(g0[i].flatten(), d[i].flatten())

            for i in range(len(g0_b)):
                gradi_product += np.dot(g0_b[i].flatten(), db[i].flatten())


            F1 = F0 + epsk * gradi_product

            y0[k] = np.abs(Fk - F0)
            y1[k] = np.abs(Fk - F1)

            nn.layers_W = copy.deepcopy(copy_layers_W)
            nn.layers_b = copy.deepcopy(copy_layers_b)

        plt.semilogy(y0)
        plt.semilogy(y1)

        plt.xlabel("Iterations")
        plt.ylabel("error")
        plt.title("Gradient Test - Forward vs Backward - Standard NN")
        plt.legend(["zero order", "first order"])  # first order should be is steeper
        plt.show()
    #endregion

    #region ResNet
    @staticmethod
    def gradient_test_W1_ResNet(Func, jacobian):
        """
        successful gradient test is when the first order is steeper than the zero order
        """
        n = 20
        k = 10
        samples = 13
        x = random.rand(n, samples)
        w = random.rand(k, n)  # (k x n)
        w2 = random.rand(n, k)  # (n x k)

        d = random.rand(k, n)  # size of w
        u = random.rand(n, samples)
        # y = random.randint(2, size=n) # size of v
        epsilon = 0.1
        F0 = Func(w, x, u, w2, 0)
        g0 = jacobian(w, x, u, w2, 0)
        iterations = 8
        y0 = np.zeros(iterations)
        y1 = np.zeros(iterations)

        for k in range(iterations):
            epsk = epsilon * pow(0.5, k)
            Fk = Func(w + epsk * d, x, u, w2, 0)
            F1 = F0 + epsk * np.dot(g0.flatten(), d.flatten())
            y0[k] = np.abs(Fk - F0)
            y1[k] = np.abs(Fk - F1)

        plt.semilogy(y0)
        plt.semilogy(y1)

        plt.xlabel("Iterations")
        plt.ylabel("error")
        plt.title("Jacobian Transposed Test - W1 - ResNet")
        plt.legend(["zero order", "first order"])  # first order should be is steeper
        plt.show()

    @staticmethod
    def gradient_test_W2_ResNet(Func, jacobian):
        """
        successful gradient test is when the first order is steeper than the zero order
        """
        n = 20
        k = 10
        samples = 2
        x = random.rand(n, samples)
        w = random.rand(k, n)  # (k x n)
        w2 = random.rand(n, k)  # (n x k)
        d = random.rand(n, k)  # size of w
        u = random.rand(n, samples)
        # y = random.randint(2, size=n) # size of v
        epsilon = 0.1
        F0 = Func(w, x, u, w2, 0)
        g0 = jacobian(w, x, u, 0)
        iterations = 8
        y0 = np.zeros(iterations)
        y1 = np.zeros(iterations)

        for k in range(iterations):
            epsk = epsilon * pow(0.5, k)
            Fk = Func(w, x, u, w2 + epsk * d, 0)
            F1 = F0 + epsk * np.dot(g0.flatten(), d.flatten())
            y0[k] = np.abs(Fk - F0)
            y1[k] = np.abs(Fk - F1)

        plt.semilogy(y0)
        plt.semilogy(y1)

        plt.xlabel("Iterations")
        plt.ylabel("error")
        plt.title("Jacobian Transposed Test - W2 - ResNet")
        plt.legend(["zero order", "first order"])  # first order should be is steeper
        plt.show()

    @staticmethod
    def gradient_test_X_ResNet(Func, jacobian):
        """
        successful gradient test is when the first order is steeper than the zero order
        """
        n = 20
        k = 10
        samples = 20
        x = random.rand(n, samples)
        w = random.rand(k, n)  # (k x n)
        w2 = random.rand(n, k)  # (n x k)

        d = random.rand(n, samples)  # size of x
        u = random.rand(n, samples)
        # y = random.randint(2, size=n) # size of v
        epsilon = 0.1
        F0 = Func(w, x, u, w2, 0)
        g0 = jacobian(w, x, u, w2)
        iterations = 20
        y0 = np.zeros(iterations)
        y1 = np.zeros(iterations)

        for k in range(iterations):
            epsk = epsilon * pow(0.5, k)
            Fk = Func(w, x + epsk * d, u, w2, 0)
            F1 = F0 + epsk * np.dot(g0.flatten(), d.flatten())
            y0[k] = np.abs(Fk - F0)
            y1[k] = np.abs(Fk - F1)

        plt.semilogy(y0)
        plt.semilogy(y1)

        plt.xlabel("Iterations")
        plt.ylabel("error")
        plt.title("Jacobian Transposed Test - X - ResNet")
        plt.legend(["zero order", "first order"])  # first order should be is steeper
        plt.show()

    @staticmethod
    def gradient_test_b_ResNet(Func, jacobian):
        """
        successful gradient test is when the first order is steeper than the zero order
        """
        n = 20
        k = 10
        samples = 20
        x = random.rand(n, samples)
        w = random.rand(k, n)  # (k x n)
        w2 = random.rand(n, k)  # (n x k)
        b = random.rand(k, 1)
        d = random.rand(k, 1)  # size of x
        u = random.rand(n, samples)
        # y = random.randint(2, size=n) # size of v
        epsilon = 0.1
        F0 = Func(w, x, u, w2, b)
        g0 = jacobian(w, x, u, w2, b)
        iterations = 8
        y0 = np.zeros(iterations)
        y1 = np.zeros(iterations)

        for k in range(iterations):
            epsk = epsilon * pow(0.5, k)
            Fk = Func(w, x , u, w2, b + epsk * d)
            F1 = F0 + epsk * np.dot(g0.flatten(), d.flatten())
            y0[k] = np.abs(Fk - F0)
            y1[k] = np.abs(Fk - F1)

        plt.semilogy(y0)
        plt.semilogy(y1)

        plt.xlabel("Iterations")
        plt.ylabel("error")
        plt.title("Jacobian Transposed Test - b - ResNet")
        plt.legend(["zero order", "first order"])  # first order should be is steeper
        plt.show()

    @staticmethod
    def direct_jacobian_transposed_test_ResNet():
        """
        successful jacobian test is when the first order is steeper than the zero order
        """
        activation = ActivationFunction.tanh
        f = ForwardActivationFunction.get_forward_activation(activation, ResNet=True)  # forward

        # g = lambda w1, x, v, w2: np.inner(np.squeeze(f(w1, x, w2)).flatten(), np.squeeze(v).flatten())  # only for test
        def g(w1, x, v, w2, b):
            return np.inner(np.squeeze(f(w1, x, w2, b)).flatten(), np.squeeze(v).flatten())

        Jaco_W1, Jaco_W2 = get_Jacobian_W(activation, ResNet=True)
        Jaco_X = get_Jacobian_X(activation, ResNet=True)
        Jaco_b = get_Jacobian_b(activation, ResNet=True)
        GradientTest.gradient_test_W2_ResNet(g, Jaco_W2)
        GradientTest.gradient_test_W1_ResNet(g, Jaco_W1)
        GradientTest.gradient_test_X_ResNet(g, Jaco_X)
        GradientTest.gradient_test_b_ResNet(g, Jaco_b)

    @staticmethod
    def gradient_test_ResNet_nn():
        """
        """
        classes = 5
        input_features = 4
        res_net_transform_size = 16
        samples = 10
        epsilon = 0.1

        nn = NeuralNetwork(input_features, [5, 3], classes, "tanh", ResNet=True,resnet_size=res_net_transform_size)

        x = random.randn(input_features, samples)
        y = random.randint(classes, size=samples)
        C = getCMatrix(y, nn.num_of_classes).T

        d1 = [np.random.randn(res_net_transform_size, input_features), np.random.randn(5, res_net_transform_size),
              np.random.randn(3, res_net_transform_size),
              np.random.randn(classes, res_net_transform_size)]
        d2 = [np.random.randn(input_features, res_net_transform_size), np.random.randn(res_net_transform_size, 5),
              np.random.randn(res_net_transform_size, 3)]

        db = [np.random.randn(res_net_transform_size, 1), np.random.randn(5, 1), np.random.randn(3, 1),
              np.random.randn(classes, 1)]

        nn.forward(x)

        F0 = F(nn.x_layers[-2], nn.layers_W[-1].T, C)
        g0 = nn.backward(y)
        iterations = 16
        y0 = np.zeros(iterations)
        y1 = np.zeros(iterations)

        for k in range(iterations):
            epsk = epsilon * pow(0.5, k)
            copy_layers_W = copy.deepcopy(nn.layers_W)
            copy_layers_W2 = copy.deepcopy(nn.layers_W2)
            copy_layers_b = copy.deepcopy(nn.layers_b)

            for i in range(len(d1)):
                print(d1[i].shape)
                nn.layers_W[i] += epsk * d1[i]

            for i in range(len(d2)):
                nn.layers_W2[i] += epsk * d2[i]

            for i in range(len(db)):
                nn.layers_b[i] += epsk * db[i]

            nn.forward(x)

            Fk = F(nn.x_layers[-2], nn.layers_W[-1].T, C)
            gradi_product = 0

            g0W1 = g0[0]
            g0W2 = g0[1]
            g0b = g0[2]
            for i in range(len(g0W1)):
                gradi_product += np.dot(g0W1[i].flatten(), d1[i].flatten())

            for i in range(len(g0W2)):
                gradi_product += np.dot(g0W2[i].flatten(), d2[i].flatten())

            for i in range(len(g0b)):
                gradi_product += np.dot(g0b[i].flatten(), db[i].flatten())

            F1 = F0 + epsk * gradi_product

            y0[k] = np.abs(Fk - F0)
            y1[k] = np.abs(Fk - F1)

            nn.layers_W = copy.deepcopy(copy_layers_W)
            nn.layers_W2 = copy.deepcopy(copy_layers_W2)
            nn.layers_b = copy.deepcopy(copy_layers_b)
        plt.semilogy(y0)
        plt.semilogy(y1)

        plt.xlabel("Iterations")
        plt.ylabel("error")
        plt.title("Gradient Test - Forward vs Backward - ResNet")
        plt.legend(["zero order", "first order"])
        plt.show()
    # endregion


def data_problem(file_path, title, alpha, epochs, batch_size, delta,input_features,output_classes,alpha_decay=1.0,useResNet = False):
    data = scipy.io.loadmat(file_path)
    X_train = data['Yt']
    C_train = data['Ct']
    # convert y_train from 1 hot encoding matrix to vector
    y_train = np.argmax(C_train, axis=0)
    X_validation = data['Yv']
    C_validation = data['Cv']
    y_validation = np.argmax(C_validation, axis=0)
    # plot the data
    plt.scatter(X_train.T[:, 0], X_train.T[:, 1], c=y_train, label='Train')

    # plt.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap='viridis', marker='s', label='Validation')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar(label='Class')
    plt.legend()
    plt.show()

    # print dimensions of data:
    print("X_train shape: ", X_train.shape)
    print("C_train shape: ", C_train.shape)
    print("y_train shape: ", y_train.shape)

    nn = NeuralNetwork(input_features, [20,5,20,5], output_classes, "relu", ResNet=useResNet)
    draw_accuracy_graph(nn, X_train, y_train, X_validation, y_validation, epochs,batch_size,alpha,
                        title + ", lr=" + str(alpha) + ", epochs=" + str(epochs) + ", batch size=" + str(batch_size)+ (", lr decay="+str(alpha_decay) if alpha_decay!=1.0 else ""),alpha_decay)


def draw_accuracy_graph(nn: NeuralNetwork, X_train, y_train, X_test, y_test, fit_epochs,batch_size,alpha, title,alpha_decay=1):
    loss_train = []
    accuracy_train = []
    loss_test = []
    accuracy_test = []
    for i in range(fit_epochs):
        nn.fit(X_train, y_train,alpha, 1, 1, batch_size)
        if i % 100 == 99 and (alpha_decay != 1):
            alpha *= alpha_decay
            print("decaying alpha", alpha)

        predict_Y = nn.predict(X_train)
        accuracy = np.sum(predict_Y == y_train) / len(X_train.T)
        accuracy_train.append(accuracy)
        loss = F(nn.x_layers[-2], nn.layers_W[-1].T, getCMatrix(y_train, nn.num_of_classes).T)
        print("loss train: ", loss)
        loss_train.append(loss)
        predict_Y = nn.predict(X_test)
        accuracy = np.sum(predict_Y == y_test) / len(X_test.T)
        accuracy_test.append(accuracy)
        loss = F(nn.x_layers[-2], nn.layers_W[-1].T, getCMatrix(y_test, nn.num_of_classes).T)
        print("loss test: ", loss)
        loss_test.append(loss)
    plt.plot(loss_train)
    plt.plot(loss_test)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend(["Train loss", "Test loss"])
    plt.subplots_adjust(bottom=0.3)  # create space at bottom
    plt.text(0 , -0.2, "ResNet" if nn.isResNet else "Standard NN", ha='left', va='center', transform=plt.gca().transAxes)
    plt.text(0, -0.3, str(nn), ha='left', va='center', transform=plt.gca().transAxes)

    plt.show()
    plt.plot(accuracy_train)
    plt.plot(accuracy_test)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend(["Train accuracy", "Test accuracy"])
    plt.subplots_adjust(bottom=0.3)  # create space at bottom

    plt.text(0, -0.2, "ResNet" if nn.isResNet else "Standard NN", ha='left', va='center', transform=plt.gca().transAxes)
    plt.text(0, -0.3, str(nn), ha='left', va='center', transform=plt.gca().transAxes)

    plt.show()


def test_data():
    alpha = 0.5
    epochs = 120
    batch_size = 75
    lr_decay = 0.5
    useResNet = True
    # data_problem('./datasets/SwissRollData.mat', "Swiss Roll", alpha, epochs, batch_size, 0.0001)
    data_problem('./datasets/GMMData.mat', "GMM", alpha, epochs, batch_size, 0.0001,5,5,alpha_decay=lr_decay,useResNet=useResNet)
    # data_problem('./datasets/PeaksData.mat', "Peaks", alpha, epochs, batch_size, 0.0001,2,5,alpha_decay=lr_decay,useResNet=useResNet)


test_data()

# GradientTest.gradient_test_FC_nn()
# GradientTest.gradient_test_ResNet_nn()
# GradientTest.direct_jacobian_transposed_test_NN()
# GradientTest.direct_jacobian_transposed_test_ResNet()
