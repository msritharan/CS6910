import numpy as np
import pandas as pd

class FeedforwardNN:
    def __init__(self, weight_init, num_layers, hidden_size, activation, input_size, output_size):
        # all hidden layers have the same size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # weights between layers
        self.weights = []
        self.weights.append(np.zeros(input_size, hidden_size))
        layer_id = 1
        while(layer_id < num_layers):
            self.weights.append(np.zeros(hidden_size, hidden_size))
        self.weights.append(np.zeros(hidden_size, output_size))

        # initialize weights as per 'weight_init'
        # uniform random and normalized xavier
        if(weight_init == "random"):
            for idx in range(len(self.weights)):
                self.weights[idx] = np.random.uniform(low = -0.2, high = 0.2, size = self.weights[idx].shape)
        elif(weight_init == "Xavier"):
            for idx in range(len(self.weights)):
                d0, d1 = self.weights[idx].shape
                self.weights[idx] = np.sqrt(1/d0)*np.random.randn(d0, d1)

        # bias of layers
        # can bias be initialized with zeros (?)
        self.bias = np.zeros(num_layers + 1)

    def activate_layer(self, a):
        if self.activation == "identity":
            return a
        if self.activation == "sigmoid":
            return self.sigmoid(a)
        elif self.activation == "tanh":
            return self.tanh(a)
        elif self.activation == "ReLU":
            return self.ReLU(a)
        
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    def ReLU(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        return np.exp(x)/sum(np.exp(x))
    
    def forward_propagate(self, input):
        h = input
        # computes h till the last hidden layer
        for idx in range(len(self.weights) - 1):
            # compute pre activated output 'a'
            a = np.matmul(self.weights[idx].T, h)
            h = self.activate_layer(a)
        
        # output layer computation
        idx = len(self.weights) - 1
        a = np.matmul(self.weights[idx], h)
        h = self.softmax(a)

        return h
    


