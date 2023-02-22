import numpy as np
from neural_network import *
NNmodel = FeedforwardNN()

def sgd(model, learning_rate, batch_size, epochs, input, label):
    W = NNmodel.weights
    b = NNmodel.bias
    a, h, y = NNmodel.forward_propagate(input)
    grad_W, grad_b = NNmodel.back_propagation(a, h, y, label)
    return grad_W, grad_b




        