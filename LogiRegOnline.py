# -*- coding: utf-8 -*-
"""


@author: Swati
"""

import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self, alpha=0.001, epochs=100000):
        self.alpha = alpha
        self.weights = np.ones(3)
        self.epochs = epochs
        self.loss_threshold = 0.001
        self.current_loss = float('inf')
        self.previous_loss = float('inf')
        self.training_converged = False
        self.iteration_count = 0
        self.tot_iter = []
        self.norm_current_loss = []

    def activ_sig(self, z):
        return 1 / (1 + np.exp(-z))

    def cross_entropy(self, predicted, actual):
        return (-actual * np.log(predicted) - (1 - actual) * np.log(1 - predicted)).mean()

    def learn_online(self, X, y):
        net_val = np.dot(X, self.weights)
        pred = self.activ_sig(net_val)
        self.gradient = np.dot(X.T, (pred - y))
        self.weights -= self.alpha * self.gradient
        self.current_loss = self.cross_entropy(pred, y)

        if self.iteration_count % 100 == 0:
            print("Cross - Entropy loss at iteration %s: %s" % (self.iteration_count + 1, self.current_loss))

        if (self.gradient == np.zeros(X.shape[0])).all():
            print("Gradient is zero!")
            print("total no. of iterations run: ", self.iteration_count + 1)
            self.training_converged = True

        if self.current_loss < self.loss_threshold:
            print("Loss optimized is less than threshold!")
            print("total no. of iterations run: ", self.iteration_count + 1)
            self.training_converged = True

        self.norm_current_loss.append(abs(self.current_loss - self.previous_loss))

        self.previous_loss = self.current_loss
        self.iteration_count += 1
        self.store_iter = self.iteration_count
        self.tot_iter.append(self.store_iter)
        # print ("iter",self.tot_iter)
        return self.norm_current_loss, self.tot_iter

    def learn(self, X, target):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        for epoch in range(self.epochs):
            print("\n\nEpoch - ", epoch + 1)
            for i in range(X.shape[0]):
                self.learn_online(X[i, :], target[i])
                if self.training_converged:
                    break

            if self.training_converged:
                break

    def predict(self, X):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return np.round(self.activ_sig(np.dot(X, self.weights)))

    def calc_cost(self, theta, X, y):
        # cost -- negative log-likelihood cost for logistic regression
        m = X.shape[0]
        y_hat = self.activ_sig(np.dot(X, theta))
        cost = (-1 / m) * (np.dot(y.T, np.log(y_hat)) + np.dot((1 - y).T, np.log(1 - y_hat)))
        return cost

    def gradient_descent(self, theta, X, y):
        m = X.shape[0]
        y_hat = self.activ_sig(np.dot(X, theta))
        dtheta = (1 / m) * (np.dot(X.T, (y_hat - y)))
        return dtheta

    def logistic_regression(self, X, y, learning_rate, iterr):
        # weights initialization
        theta = np.zeros(X.shape[1]).reshape(2, 1)
        costs = []
        for i in range(iterr):
            cost = self.calc_cost(theta, X, y)
            dtheta = self.gradient_descent(theta, X, y)
            # Updating weights and cost
            theta = theta - learning_rate * dtheta
            costs.append(cost[0, 0])

        return costs, theta, dtheta


def generate_data(mean, variance, count):
    return np.random.multivariate_normal(mean, variance, count)


def calcAcc(pred_y, test_y):
    pred_y = pred_y.tolist()
    test_y = test_y.tolist()

    count = 0
    for i in range(len(pred_y)):
        if pred_y[i] == test_y[i]:
            count += 1

    return (count / len(pred_y)) * 100


max_epochs = 100000
x1 = generate_data([1, 0], [[1, 0.75], [0.75, 1]], 500)
x2 = generate_data([0, 1.5], [[1, 0.75], [0.75, 1]], 500)
X = np.vstack((x1, x2)).astype(np.float32)
y = np.hstack((np.zeros(500), np.ones(500)))

test_x1 = generate_data([1, 0], [[1, 0.75], [0.75, 1]], 500)
test_x2 = generate_data([0, 1.5], [[1, 0.75], [0.75, 1]], 500)
test_X = np.vstack((test_x1, test_x2)).astype(np.float32)
test_y = np.hstack((np.zeros(500), np.ones(500)))

print("\n\nLearning rate (Alpha): 1\nTotal Epochs: 100000")
lr = LogisticRegression(alpha=1, epochs=max_epochs)
lr.learn(X, y)
print("Final Weights: ", lr.weights)
predicted_y = lr.predict(test_X)
accuracy = calcAcc(predicted_y, test_y)
print("Accuracy: ", accuracy)
f, ax = plt.subplots(1, figsize=(5, 5))
ax.set_title("Norm Gradient")
ax.plot(lr.tot_iter, lr.norm_current_loss, 'r', label=r'$\alpha = 1$')
ax.set_xlabel('Iterations')
ax.set_ylabel('Gradient')
ax.legend();
print("==================================================")

print("\n\nLearning rate (Alpha): 0.1\nTotal Epochs: 100000")
lr = LogisticRegression(alpha=0.1, epochs=max_epochs)
lr.learn(X, y)
print("Final Weights: ", lr.weights)
predicted_y = lr.predict(test_X)
accuracy = calcAcc(predicted_y, test_y)
print("Accuracy: ", accuracy)
f, ax = plt.subplots(1, figsize=(5, 5))
ax.set_title("Norm Gradient")
ax.plot(lr.tot_iter, lr.norm_current_loss, 'r', label=r'$\alpha = 0.1$')
ax.set_xlabel('Iterations')
ax.set_ylabel('Gradient')
ax.legend();
print("==================================================")

print("\n\nLearning rate (Alpha): 0.01\nTotal Epochs: 100000")
lr = LogisticRegression(alpha=0.01, epochs=max_epochs)
lr.learn(X, y)
print("Final Weights: ", lr.weights)
predicted_y = lr.predict(test_X)
accuracy = calcAcc(predicted_y, test_y)
print("Accuracy: ", accuracy)
f, ax = plt.subplots(1, figsize=(5, 5))
ax.set_title("Norm Gradient")
ax.plot(lr.tot_iter, lr.norm_current_loss, 'r', label=r'$\alpha = 0.01$')
ax.set_xlabel('Iterations')
ax.set_ylabel('Gradient')
ax.legend();
print("==================================================")

print("\n\nLearning rate (Alpha): 0.001\nTotal Epochs: 100000")
lr = LogisticRegression(alpha=0.001, epochs=max_epochs)
lr.learn(X, y)
print("Final Weights: ", lr.weights)
predicted_y = lr.predict(test_X)
accuracy = calcAcc(predicted_y, test_y)
print("Accuracy: ", accuracy)
f, ax = plt.subplots(1, figsize=(5, 5))
ax.set_title("Norm Gradient")
ax.plot(lr.tot_iter, lr.norm_current_loss, 'r', label=r'$\alpha = 0.001$')
ax.set_xlabel('Iterations')
ax.set_ylabel('Gradient')
ax.legend();
print("==================================================")

plt.figure(figsize=(5, 8))
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=.4)
plt.show()

y1 = y.reshape(X.shape[0], 1)
learning_rate = [1, 0.1, 0.01, 0.001]
for i in learning_rate:
    costs, theta, dtheta = lr.logistic_regression(X, y1, learning_rate=i, iterr=100000)
    f, ax = plt.subplots(1, figsize=(5, 5))
    indices = np.float32(range(100000))
    ax.set_title("Validation loss")
    ax.plot(indices, costs, 'r', label=r'$\alpha = {0}$'.format(i))
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Training Error')
    ax.legend();
plt.figure(figsize=(5, 8))
x_d = np.linspace(-3, 4, 50)
y_d = -(theta[0] * x_d) / theta[1]
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=.4)
plt.plot(x_d, y_d)
plt.show()