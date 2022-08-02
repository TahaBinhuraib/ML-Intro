import matplotlib.pyplot as plt
import numpy as np
import torch


def normalize(X_train: np.array, X_test: np.array):
    # Normalize (mean zero, unit variance)
    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    return X_train, X_test


def plot(X_train: np.array, y_train: np.array):
    plt.scatter(
        X_train[y_train == 0, 0], X_train[y_train == 0, 1], label="class 0", marker="o"
    )
    plt.scatter(
        X_train[y_train == 1, 0], X_train[y_train == 1, 1], label="class 1", marker="s"
    )
    plt.title("Training set")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.xlim([-3, 4])
    plt.ylim([-3, 3])
    plt.legend()
    plt.show()


data = np.genfromtxt("perceptron_toydata.txt", delimiter="\t")
X, y = data[:, :2], data[:, 2]
y = y.astype(np.int)

print("Class label counts:", np.bincount(y))
print("X.shape:", X.shape)
print("y.shape:", y.shape)

# Shuffling & train/test split
shuffle_idx = np.arange(y.shape[0])
shuffle_rng = np.random.RandomState(123)
shuffle_rng.shuffle(shuffle_idx)
X, y = X[shuffle_idx], y[shuffle_idx]

X_train, X_test = X[shuffle_idx[:70]], X[shuffle_idx[70:]]
y_train, y_test = y[shuffle_idx[:70]], y[shuffle_idx[70:]]

X_train, X_test = normalize(X_train, X_test)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is: {device}")


class Perceptron:
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = torch.zeros(num_features, 1, dtype=torch.float32, device=device)
        self.bias = torch.zeros(1, dtype=torch.float32, device=device)

        # placeholder vectors so they don't
        # need to be recreated each time
        self.ones = torch.ones(1)
        self.zeros = torch.zeros(1)

    def forward(self, x):
        linear = torch.mm(x, self.weights) + self.bias
        predictions = torch.where(linear > 0.0, self.ones, self.zeros)
        return predictions

    def backward(self, x, y):
        predictions = self.forward(x)
        errors = y - predictions
        return errors

    def train(self, x, y, epochs):
        for e in range(epochs):

            for i in range(y.shape[0]):
                # use view because backward expects a matrix (i.e., 2D tensor)
                errors = self.backward(
                    x[i].reshape(1, self.num_features), y[i]
                ).reshape(-1)
                self.weights += (errors * x[i]).reshape(self.num_features, 1)
                self.bias += errors

    def evaluate(self, x, y):
        predictions = self.forward(x).reshape(-1)
        accuracy = torch.sum(predictions == y).float() / y.shape[0]
        return accuracy


perceptron = Perceptron(num_features=2)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)

perceptron.train(X_train_tensor, y_train_tensor, epochs=5)

print("Model parameters:")
print(f"Weights: {perceptron.weights}")
print(f"Bias: {perceptron.bias}")


X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)

test_acc = perceptron.evaluate(X_test_tensor, y_test_tensor)
print("Test set accuracy: %.2f%%" % (test_acc * 100))
