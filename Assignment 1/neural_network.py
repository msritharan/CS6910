import numpy as np

class FeedforwardNN:
    def __init__(self, weight_init, num_layers, hidden_size, activation, input_size, output_size):
        # all hidden layers have the same size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # weights between layers
        self.weights = []
        self.weights.append(np.zeros(hidden_size, input_size))
        layer_id = 1
        while(layer_id < num_layers):
            self.weights.append(np.zeros(hidden_size, hidden_size))
        self.weights.append(np.zeros(output_size, hidden_size))

        # bias for each layers
        self.bias = []
        for L in range(num_layers):
            self.bias.append(np.zeros(hidden_size))
        self.bias.append(np.zeros(output_size))

        # initialize weights nad bias as per 'weight_init'
        # uniform random and normalized xavier
        # bias is initialized to zeros 
        if(weight_init == "random"):
            for idx in range(len(self.weights)):
                self.weights[idx] = np.random.uniform(low = -0.2, high = 0.2, size = self.weights[idx].shape)
        elif(weight_init == "Xavier"):
            for idx in range(len(self.weights)):
                d0, d1 = self.weights[idx].shape
                self.weights[idx] = np.sqrt(1/d0)*np.random.randn(d0, d1)


    def activate_layer(self, a):
        if self.activation == "identity":
            return self.identity(a)
        elif self.activation == "sigmoid":
            return self.sigmoid(a)
        elif self.activation == "tanh":
            return self.tanh(a)
        elif self.activation == "ReLU":
            return self.ReLU(a)
    
    # activation functions
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    def ReLU(self, x):
        return np.maximum(0, x)

    def identity(self, x):
        return x
    
    def softmax(self, x):
        return np.exp(x)/sum(np.exp(x))
    
    # gradients of activation functions
    def grad_sigmoid(self, x):
        return self.sigmoid(x)*(1 - self.sigmoid(x))
    
    def grad_tanh(self, x):
        return 1 - (self.tanh(x))**2
    
    def grad_ReLU(self, x):
        return (x >= 0)*1
    
    def grad_identity(self, x):
        return 1
    
    def grad_softmax(self, x):
        return np.diag(x) - np.outer(x, x)

    def grad_activate(self, x):
        if self.activation == "identity":
            return self.grad_identity(x)
        elif self.activation == "sigmoid":
            return self.grad_sigmoid(x)
        elif self.activation == "tanh":
            return self.grad_tanh(x)
        elif self.activation == "ReLU":
            return self.grad_ReLU(x)
        
    def forward_propagate(self, input):
        h = input
        a_layer = [] # pre activation outputs of each layer
        h_layer = [] # activated outputs of each layer
        
        for l in range(len(self.weights)):
            a = self.bias[l] + self.weights[l]*h
            h = self.activate_layer(a)
            a_layer.append(a)
            h_layer.append(h)
        
        L = len(self.weights) - 1
        a = self.bias[L] + self.weights[L]*h
        h = self.softmax(a) 
        a_layer.append(a)
        h_layer.append(h) 

        return a_layer, h_layer, h
    
    def back_propagation(self, a_layer, h_layer, ypred, ylabel):
        grad_W = [] #   list of gradients wrt weights between layers
        grad_b = [] #   list of gradients wrt biases
        L = len(h_layer) - 1

        # gradient wrt to a_L
        e_y = np.zeros(self.output_size)
        e_y[ylabel] = 1

        # propagate backwards
        g_a = -(e_y - ypred)
        for l in range(L, -1, -1):
            # gradients wrt weights and bias
            g_Wl = np.outer(g_a, h_layer[l])
            g_bl = g_a
            grad_W.append(g_Wl)
            grad_b.append(g_bl)

            #  gradients wrt layer below
            g_h = self.weights[l].T*g_a
            g_a = g_h*self.grad_activate(a_layer[l])

        grad_W.reverse()
        grad_b.reverse()

        return grad_W, grad_b




