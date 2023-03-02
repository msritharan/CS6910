from get_inputs import *
from neural_network import *
from optimizers import *

import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import fashion_mnist

# load dataset and process it
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1)/255
x_test = x_test.reshape(x_test.shape[0], -1)/255

# split train into train + val
train_val_split = 0.9
indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)
total_samples = x_train.shape[0]
Xtrain = np.array([x_train[idx] for idx in range(int(train_val_split*total_samples))])
Ytrain = np.array([y_train[idx] for idx in range(int(train_val_split*total_samples))])
Xval = np.array([x_train[idx] for idx in range(int(train_val_split*total_samples), total_samples)])
Yval = np.array([y_train[idx] for idx in range(int(train_val_split*total_samples), total_samples)])
Xtest = x_test
Ytest = y_test

print("Train Data Dimensions : ", Xtrain.shape)
print("Validation Data Dimensions :", Xval.shape)
print("Test Data Dimensions :", Xtest.shape)

# write training loop for given inputs
input_size = 28*28
output_size = 10
model = FeedforwardNN(weight_init, num_layers, hidden_size, activation, input_size, output_size, loss_function)
model.weights, model.bias, train_acc, train_loss, val_acc, val_loss = train_model(model, optimizer, learning_rate, batch_size, epochs, momentum, beta, beta1, beta2, epsilon, Xtrain, Ytrain, Xval, Yval)
# train_loss /= Xtrain.shape[0]
# val_loss /= Xval.shape[0]

# Visualize training and validation metrics
plt.figure()
plt.plot(train_acc, c = 'r', label = 'train_acc')
plt.plot(val_acc, c = 'g', label = 'val_acc')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.title('Train and Val Accuracies')
plt.show()

plt.figure()
plt.plot(train_loss, c = 'r', label = 'train_loss')
plt.plot(val_loss, c = 'b', label = 'val_loss')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.title('Train and Val Losses')
plt.show()

# evaluate model on test data
test_acc, test_loss = model.evaluate_metrics(Xtest, Ytest)
#test_loss /= Xtest.shape[0]
print("Test Accuracy and Loss : ", test_acc, test_loss)