from get_inputs import *
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import fashion_mnist

# Q1 - load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
plt.figure()
plt.imshow(x_train[0], cmap = 'gray')
plt.show()