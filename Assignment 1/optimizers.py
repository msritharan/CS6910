import numpy as np
import matplotlib.pyplot as plt
# from neural_network import *
# NNmodel = FeedforwardNN()

def sgd(NNmodel, learning_rate, batch_size, epochs, X, Y, Xval, Yval):
    iter_test_acc = []
    for iter in range(epochs):
        
        W = NNmodel.weights
        b = NNmodel.bias
        gW = 0
        gb = 0

        sample_indices = np.arange(len(X))
        np.random.shuffle(sample_indices)

        for idx in range(len(X)):
            a, h, y = NNmodel.forward_propagate(X[sample_indices[idx]])
            grad_W, grad_b = NNmodel.back_propagation(a, h, y, Y[sample_indices[idx]])
            if gW == 0 and gb == 0:
                gW = grad_W
                gb = grad_b
            else:
                for idx2 in range(len(grad_W)):
                    gW[idx2] += grad_W[idx2]
                    gb[idx2] += grad_b[idx2]

            # end of a batch or end of array
            if ((idx + 1)%batch_size == 0) or (idx == len(X) - 1):
                for idx2 in range(len(W)):
                    W[idx2] -= learning_rate*gW[idx2]
                    b[idx2] -= learning_rate*gb[idx2]

                NNmodel.weights = W
                NNmodel.bias = b
                gW = 0
                gb = 0
        
        epoch_acc = NNmodel.evaluate(Xval, Yval)
        print("Epoch : ", iter, "Accuracy : ", epoch_acc*100)
        iter_test_acc.append(epoch_acc)
    
    plt.figure()
    plt.plot(iter_test_acc)
    plt.show()
    
    return NNmodel.weights, NNmodel.bias

def momentum_sgd(NNmodel, learning_rate, batch_size, epochs, momentum, X, Y, Xval, Yval):
    iter_test_acc = []
    for iter in range(epochs):
        
        W = NNmodel.weights
        b = NNmodel.bias
        gW = 0
        gb = 0

        sample_indices = np.arange(len(X))
        np.random.shuffle(sample_indices)
        hist_update_W = 0
        hist_update_b = 0
        for idx in range(len(X)):
            a, h, y = NNmodel.forward_propagate(X[sample_indices[idx]])
            grad_W, grad_b = NNmodel.back_propagation(a, h, y, Y[sample_indices[idx]])
            if gW == 0 and gb == 0:
                gW = grad_W
                gb = grad_b
            else:
                for idx2 in range(len(grad_W)):
                    gW[idx2] += grad_W[idx2]
                    gb[idx2] += grad_b[idx2]

            # end of a batch or end of array
            if ((idx + 1)%batch_size == 0) or (idx == len(X) - 1):
                if(hist_update_W == 0 and hist_update_b == 0):
                    update_W = []
                    update_b = []
                    for idx2 in range(len(W)):
                        update_W.append(gW[idx2])
                        update_b.append(gb[idx2])
                        W[idx2] -= learning_rate*update_W[idx2]
                        b[idx2] -= learning_rate*update_b[idx2]
                    
                    hist_update_W = update_W
                    hist_update_b = update_b
                else:
                    update_W = []
                    update_b = []
                    for idx2 in range(len(W)):
                        update_W.append(momentum*hist_update_W[idx2] + gW[idx2])
                        update_b.append(momentum*hist_update_b[idx2] + gb[idx2])
                        W[idx2] -= learning_rate*update_W[idx2]
                        b[idx2] -= learning_rate*update_b[idx2]

                    hist_update_W = update_W
                    hist_update_b = update_b

                NNmodel.weights = W
                NNmodel.bias = b
                gW = 0
                gb = 0
        
        epoch_acc = NNmodel.evaluate(Xval, Yval)
        print("Epoch : ", iter, "Accuracy : ", epoch_acc*100)
        iter_test_acc.append(epoch_acc)
    
    plt.figure()
    plt.plot(iter_test_acc)
    plt.show()
    
    return NNmodel.weights, NNmodel.bias
