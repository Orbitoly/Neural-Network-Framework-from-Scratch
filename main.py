import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import scipy.io

np.random.seed(20)

"""
Target function and its gradients
"""
def F(X, W, C):
    """
    :return: calculates the F for all the classes
    """
    assert X.shape[1] == C.shape[0], "Dimensions mismatch: X and C"
    assert X.shape[0] == W.shape[0], "Dimensions mismatch: X and W"
    assert C.shape[1] == W.shape[1], "Dimensions mismatch: C and W"

    classes = range(W.shape[1])
    m = X.shape[1]
    W_max_row = (X.T @ W).max(axis=1)

    F_k_list = [F_class_k(X, W, k, W_max_row) for k in classes]
    rightElement = np.vstack(F_k_list)
    rightElement = rightElement.T
    return -1 * (1 / m) * (C * rightElement).sum(axis=0).sum(axis=0)


def F_class_k(X, W, k, max_row=0):
    """
    :param k: class k
    :param max_row: it is the max of the row of X.T @ W, used to avoid overflow
    :return: it calculates the F for a specific class k
    """
    temp = np.divide(np.exp(X.T @ W[:, k] - max_row), np.exp(X.T @ W - max_row.reshape(-1, 1)).sum(axis=1))
    temp = np.log(temp)

    return temp

def gradient_W(X, W, y: np.ndarray):
    classes = range(W.shape[1])
    C = getCMatrix(y, W.shape[1]).T
    WP = np.stack([gradient_wp(X, W, p, C) for p in classes])
    WP = WP.T

    return WP


def gradient_wp(X, W, p, C):
    """
    :param p: class p
    :return: it calculates the gradient for a specific class p
    """
    W_max_row = (X.T @ W).max(axis=1)

    divideLeft = np.exp(X.T @ W[:, p] - W_max_row)
    divideRight = np.exp(X.T @ W - W_max_row.reshape(-1,1)).sum(axis=1)
    rightElement = np.divide(divideLeft, divideRight)#.sum(axis=1)
    # rightElement = np.divide(np.exp(X.T @ W[:, p]), np.exp(X.T @ W).sum(axis=1))
    rightElement = rightElement - C[:, p]
    return (1 / X.shape[1]) * X @ rightElement


def gradient_X(X, W, y):
    num_classes = 1 if len(W.shape) == 1 else W.shape[1]
    C = getCMatrix(y, num_classes)
    temp = W.T @ X

    temp -= np.max(temp)
    lElementDivide = np.exp(temp)
    rElementDivide = np.exp(temp).sum(axis=0)
    # lElementDivide = np.exp(W.T @ X)
    # rElementDivide = np.exp(W.T @ X).sum(axis=0)
    division = np.divide(lElementDivide, rElementDivide)

    res = (1 / X.shape[1]) * W @ (division - C)
    return res


"""
Gradient Tests
"""

def gradient_test_W(Func, gradient):
    """
    successful gradient test is when the first order is steeper than the zero order
    """
    n = 20
    x = random.rand(n, n)
    w = random.rand(n, 2)
    d = random.rand(n, 2)  # size of w
    y = random.randint(2, size=n)
    C = getCMatrix(y, w.shape[1]).T
    epsilon = 0.1
    F0 = Func(x, w, C)
    g0 = gradient(x, w, y)

    iterations = 8
    y0 = np.zeros(iterations)
    y1 = np.zeros(iterations)

    for k in range(iterations):
        epsk = epsilon * pow(0.5, k)
        Fk = Func(x, w + epsk * d, C)
        F1 = F0 + epsk * np.dot(g0.flatten(), d.flatten())
        y0[k] = np.abs(Fk - F0)
        y1[k] = np.abs(Fk - F1)

    plt.semilogy(y0)
    plt.semilogy(y1)

    plt.xlabel("Iterations")
    plt.ylabel("error")
    plt.title("Gradient Test: F vs F_grad_W")
    plt.legend(["zero order", "first order"])  # first order should be is steeper
    plt.show()


def gradient_test_X(Func, gradient):
    n = 1
    x = random.rand(n, n)
    w = random.rand(n, 2)
    d = random.rand(n, n)  # size of x
    y = random.randint(2, size=n)
    C = getCMatrix(y, w.shape[1]).T
    epsilon = 0.1
    F0 = Func(x, w, C)
    g0 = gradient(x, w, y)

    iterations = 8
    y0 = np.zeros(iterations)
    y1 = np.zeros(iterations)
    for k in range(iterations):
        epsk = epsilon * pow(0.5, k)
        Fk = Func(x + epsk * d, w, C)
        F1 = F0 + epsk * np.dot(g0.flatten(), d.flatten())
        y0[k] = np.abs(Fk - F0)
        y1[k] = np.abs(Fk - F1)

    plt.semilogy(y0)
    plt.semilogy(y1)

    plt.xlabel("Iterations")
    plt.ylabel("error")
    plt.title("Gradient Test: F vs F_grad_X")
    plt.legend(["zero order", "first order"])  # first order should be is steeper
    plt.show()


"""
Softmax function and its accuracy calculation
"""


def calc_accuracy_softmax(X, y, W):
    predict = softmax_w(X, W)
    predict_Y = np.argmax(predict, axis=1)
    return np.sum(predict_Y == y) / len(X.T)


# returns: Cols: WCLASS / Rows: X
def softmax_w(X, W):
    return np.exp(np.dot(X.T, W)) / np.sum(np.exp(np.dot(X.T, W)), axis=1, keepdims=True)


"""
Optimization Algorithms
"""
def sgd(X, y, W, gradi, alpha, iterations, delta, batch_size):
    # TODO: use delta to stop
    W_iters = []

    num_epochs = iterations
    indexes = np.arange(X.shape[1])

    for epoch in range(num_epochs):
        if epoch % 10 == 0:
            print("Epoch:", epoch + 1)
        # Shuffle the indexes
        np.random.shuffle(indexes)
        batches = np.array_split(indexes, len(indexes) // batch_size)
        for batch in batches:
            X_batch = X[:, batch]
            y_batch = y[batch]
            g = gradi(X_batch, y_batch, W)
            g /= batch_size
            W -= alpha * g

        W_iters.append(W.copy())

    return W_iters[-1], W_iters


def sgd_momentum(X, y, W, gradi, alpha, iterations, delta, beta, batch_size):
    # TODO: use delta to stop
    W_iters = []
    v = np.zeros_like(W)
    num_epochs = iterations
    indexes = np.arange(X.shape[1])

    for epoch in range(num_epochs):
        if epoch % 100 == 0:
            print("Epoch:", epoch + 1)
        np.random.shuffle(indexes)
        batches = np.array_split(indexes, len(indexes) // batch_size)
        for batch in batches:
            X_batch = X[:, batch]
            y_batch = y[batch]
            g = gradi(X_batch, y_batch, W)
            g /= batch_size

            v = beta * v + (1 - beta) * g
            W -= alpha * v

        W_iters.append(W.copy())

    return W_iters[-1], W_iters


"""
Test Optimization Algorithms
"""


def least_squares_target_function(X, y, w):
    return np.linalg.norm(X.dot(w) - y) ** 2


def least_squares_gradient(X, y, w):
    return 2 * X @ (X.T @ w - y)


def test_sgd_with_least_squares():
    x = np.arange(0.01, 1.005, 0.005)
    a = 0.8
    b = 0.4
    epsilon = 0.2 * np.random.randn(len(x))
    y = a * x + b + epsilon
    A = np.vstack((x, np.ones(len(x))))
    B = np.copy(y)
    # opt using sgd

    # opt_ab, W_iters = sgd(A, B, np.array([0, 0], dtype=float), least_squares_gradient, alpha=0.01, iterations=1000,delta=0.0001, batch_size=8)
    opt_ab, W_iters = sgd_momentum(A, B, np.array([0, 0], dtype=float), least_squares_gradient, alpha=0.01,
                                   iterations=1000, delta=0.0001, beta=0.9, batch_size=10)

    # opt_ab = np.linalg.lstsq(A, B, rcond=None)[0]

    print("opt_ab ", opt_ab)
    print("ls ", np.linalg.lstsq(A.T, B, rcond=None)[0])
    # Alternatively, you can use np.linalg.solve(A.T @ A, A.T @ B) to compute opt_ab

    plt.close("all")
    plt.plot(x, y, ".b", label="Measurements")
    plt.plot(x, a * x + b, "-r", label="True line")
    plt.plot(x, opt_ab[0] * x + opt_ab[1], "-g", label="Estimated line using SGD")
    plt.title("Least Squares: Linear Regression Example")
    plt.legend()
    plt.show()


"""
Util functions
"""


def draw_accuracy_graph(X_train, y_train, X_test, y_test, W_iters, accuracy_func, title):
    accuracy_train = []
    accuracy_test = []
    for W in W_iters:
        accuracy_train.append(accuracy_func(X_train, y_train, W))
        accuracy_test.append(accuracy_func(X_test, y_test, W))

    plt.plot(accuracy_train)
    plt.plot(accuracy_test)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend(["Train accuracy", "Test accuracy"])
    plt.show()

def getCMatrix(labels, num_classes):
    num_samples = len(labels)
    encoded_labels = np.zeros((num_samples, num_classes))
    # class_index = {label: index for index, label in enumerate(set(labels))}

    for i, label in enumerate(labels):
        # print("label",label)
        # print("num_classes",num_classes)
        if label < num_classes:
            encoded_labels[i, label] = 1
        else:
            encoded_labels[i, -1] = 1

    encoded_labels = encoded_labels.T

    assert encoded_labels.shape == (num_classes, num_samples), "Shape of encoded labels is incorrect."

    return encoded_labels


"""
Run Data Problems
"""


def data_problem(file_path, title, alpha, iterations, batch_size, delta):
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

    # add a row of ones to X_train

    # Add bias row:
    row = np.ones((1, X_train.shape[1]), dtype=int)
    X_train = np.vstack((X_train, row))
    row = np.ones((1, X_validation.shape[1]), dtype=int)
    X_validation = np.vstack((X_validation, row))

    # plt.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap='viridis', marker='s', label='Validation')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar(label='Class')
    plt.legend()
    plt.show()

    W = np.random.rand(X_train.shape[0], C_train.shape[0])

    W, W_iter = sgd_momentum(X_train, y_train, W, gradi=gradient_W, alpha=alpha, iterations=iterations, delta=delta,
                             beta=0.9, batch_size=batch_size)
    # W, W_iter = sgd(X_train, y_train, W, gradi=gradient_W, alpha=alpha, iterations=iterations, delta=delta. batch_size=batch_size)

    draw_accuracy_graph(X_train, y_train, X_validation, y_validation, W_iter, calc_accuracy_softmax,
                        title + " alpha = " + str(alpha) + ", epochs = " + str(iterations) + ", beta = " + str(
                            0.9) + ", batch size = " + str(batch_size))

#
# def test_easy_data():
#     # Generate random datanp.random.seed(123)
#     num_samples = 1000
#     num_features = 3
#     num_classes = 3
#
#     ones_vec_num_fectures = np.ones(num_features)
#     # Generate class 0 data
#     rand_vec = np.random.randn(num_features)
#     X0 = np.random.randn(num_samples // num_classes, num_features) + 1 * rand_vec
#     y0 = np.zeros(num_samples // num_classes)
#
#     # Generate class 1 data
#     X1 = np.random.randn(num_samples // num_classes, num_features) + 3 * rand_vec
#     y1 = np.ones(num_samples // num_classes)
#
#     # Generate class 2 data
#     X2 = np.random.randn(num_samples // num_classes, num_features) + 5 * rand_vec
#     y2 = 2 * np.ones(num_samples // num_classes)
#
#     # Concatenate data from all classes
#     X = np.concatenate((X0, X1, X2), axis=0)
#     y = np.concatenate((y0, y1, y2), axis=0)
#
#     # Split data into train, validation, and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=123)
#     X_train = X_train.T
#     X_val = X_val.T
#     X_test = X_test.T
#
#     # plot the data:
#     # plt.figure(figsize=(12, 12))
#     # plt.scatter(X_train[0, :], X_train[1, :], c=y_train, cmap='viridis')
#     # plt.xlabel('Feature 1')
#     # plt.ylabel('Feature 2')
#     # plt.title('Separable Softmax Classification Problem')
#     # plt.show()
#
#     # plot the data in 3 dimensions:
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(X_train[0, :], X_train[1, :], X_train[2, :], c=y_train)
#     ax.set_title('~Separable Softmax Classification Problem')
#     plt.show()
#
#     # # Plot the data points
#     # plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', label='Train')
#     # plt.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap='viridis', marker='s', label='Validation')
#     # plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='D', label='Test')
#     # plt.xlabel('Feature 1')
#     # plt.ylabel('Feature 2')
#     # plt.title('Separable Softmax Classification Problem')
#     # plt.colorbar(label='Class')
#     # plt.legend()
#
#     alpha = 0.05
#     iterations = 1500
#     delta = 0.0001
#     batch_size = 20
#     # Add bias row:
#     row = np.ones((1, X_train.shape[1]), dtype=int)
#     X_train = np.vstack((X_train, row))
#     row = np.ones((1, X_test.shape[1]), dtype=int)
#     X_test = np.vstack((X_test, row))
#
#     W = np.random.randn(X_train.shape[0], num_classes)
#
#     print("batch size:",batch_size)
#     W, W_iter = sgd_momentum(X_train, y_train, W, gradi=gradient_W, alpha=alpha, iterations=iterations, delta=delta,
#                              beta=0.9, batch_size=batch_size)
#     # W, W_iter = sgd(X_train, y_train, W, gradi=gradient_W, alpha=alpha, iterations=iterations, delta=delta)
#
#     draw_accuracy_graph(X_train, y_train, X_test, y_test, W_iter, calc_accuracy_softmax,
#                         "Sample problem accuracy\n" + " alpha=" + str(alpha) + " iterations=" + str(
#                             iterations) + " batch_size=" + str(batch_size))
#

def main():
    # alphas = [1, 0.0001, 0.01]
    # batch_sizes = [1000]
    # iterations = 350
    #
    # for alpha in alphas:
    #     for batch_size in batch_sizes:
    #         # data_problem('./datasets/SwissRollData.mat', "Swiss Roll Data", alpha, iterations, batch_size, 0.0001)
    #         # data_problem('./datasets/GMMData.mat', "GMM Data", alpha, iterations, batch_size, 0.0001)
    #         data_problem('./datasets/PeaksData.mat', "Peaks Data", alpha, iterations, batch_size, 0.0001)
    #         pass

    # gradient_test_W(F, gradient_W)
    # test_sgd_with_least_squares()
    # test_easy_data()

    # X = np.array([[1, 1], [1, 2], [3, 5], [1, 1]]).T
    # W = np.array([[1, 0], [0, 1], [1, 1]]).T
    # y = np.array([1, 2, 1,2])
    gradient_test_X(F, gradient_X)
    # print(X.shape)
    # print(W.shape)
    # print(y.shape)
    # res = gradient_X(X, W, y)
    # print(res)
    # test_C_Matrix()
    return

def test_C_Matrix():
    y = np.array([1, 2, 1, 1])
    C = getCMatrix(y,3)
    res = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0],[0,1,0]]).T
    assert np.array_equal(C, res)

    y2 = np.array([3, 0, 1,2])
    C2 = getCMatrix(y2, 4)
    res2 = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0,0,1,0]]).T
    assert np.array_equal(C2, res2)

if __name__ == "__main__":
    main()
