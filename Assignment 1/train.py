from get_inputs import *
from neural_network import *
from optimizers import *

import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import fashion_mnist

# Q1 - load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(60000, -1)/255
x_test = x_test.reshape(x_test.shape[0], -1)/255

# plt.figure()
# plt.imshow(x_train[0], cmap = 'gray')
# plt.show()

epochs = 10

X = x_train[:10000]
Y = y_train[:10000]

# model = FeedforwardNN(weight_init, num_layers, hidden_size, activation, 28*28, 10)
# W, b = sgd(model, learning_rate, batch_size, epochs, X, Y, x_test, y_test)
# model.weights = W
# model.bias = b
# print(model.evaluate(x_test, y_test))

# model1 = FeedforwardNN(weight_init, num_layers, hidden_size, activation, 28*28, 10)
# W, b = momentum_sgd(model1, learning_rate, batch_size, epochs, momentum, X, Y, x_test, y_test)
# model1.weights = W
# model1.bias = b
# print(model1.evaluate(x_test, y_test))

# model2 = FeedforwardNN(weight_init, num_layers, hidden_size, activation, 28*28, 10)
# W, b = nag(model2, learning_rate, batch_size, epochs, momentum, X, Y, x_test, y_test)
# model2.weights = W
# model2.bias = b
# print(model2.evaluate(x_test, y_test))

# model3 = FeedforwardNN(weight_init, num_layers, hidden_size, activation, 28*28, 10)
# W, b = rmsprop(model3, learning_rate, batch_size, epochs, beta, epsilon, X, Y, x_test, y_test)
# model3.weights = W
# model3.bias = b
# print(model3.evaluate(x_test, y_test))

model4 = FeedforwardNN(weight_init, num_layers, hidden_size, activation, 28*28, 10)
W, b = adam(model4, learning_rate, batch_size, epochs, beta1, beta2, epsilon, X, Y, x_test, y_test)
model4.weights = W
model4.bias = b
print(model4.evaluate(x_test, y_test))

model5 = FeedforwardNN(weight_init, num_layers, hidden_size, activation, 28*28, 10)
W, b = nadam(model5, learning_rate, batch_size, epochs, beta1, beta2, epsilon, X, Y, x_test, y_test)
model5.weights = W
model5.bias = b
print(model5.evaluate(x_test, y_test))