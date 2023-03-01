import numpy as np
import matplotlib.pyplot as plt
# from neural_network import *
# NNmodel = FeedforwardNN()

def sgd(NNmodel, learning_rate, batch_size, epochs, X, Y, Xval, Yval):
    iter_test_acc = []
    for iter in range(epochs):
        
        W = NNmodel.weights
        b = NNmodel.bias
        gW = [np.zeros(W[idx].shape) for idx in range(len(W))]
        gb = [np.zeros(b[idx].shape) for idx in range(len(b))]

        sample_indices = np.arange(len(X))
        np.random.shuffle(sample_indices)

        for idx in range(len(X)):
            a, h, y = NNmodel.forward_propagate(X[sample_indices[idx]])
            grad_W, grad_b = NNmodel.back_propagation(a, h, y, Y[sample_indices[idx]])
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
                gW = [np.zeros(W[idx].shape) for idx in range(len(W))]
                gb = [np.zeros(b[idx].shape) for idx in range(len(b))]
        
        epoch_acc = NNmodel.evaluate(Xval, Yval)
        print("Epoch : ", iter, "Accuracy : ", epoch_acc*100)
        iter_test_acc.append(epoch_acc)
    
    # plt.figure()
    # plt.plot(iter_test_acc)
    # plt.show()
    
    return NNmodel.weights, NNmodel.bias

def momentum_sgd(NNmodel, learning_rate, batch_size, epochs, momentum, X, Y, Xval, Yval):
    iter_test_acc = []
    for iter in range(epochs):
        
        W = NNmodel.weights
        b = NNmodel.bias
        gW = [np.zeros(W[idx].shape) for idx in range(len(W))]
        gb = [np.zeros(b[idx].shape) for idx in range(len(b))]

        sample_indices = np.arange(len(X))
        np.random.shuffle(sample_indices)
        hist_update_W = [np.zeros(W[idx].shape) for idx in range(len(W))]
        hist_update_b = [np.zeros(b[idx].shape) for idx in range(len(b))]

        for idx in range(len(X)):
            a, h, y = NNmodel.forward_propagate(X[sample_indices[idx]])
            grad_W, grad_b = NNmodel.back_propagation(a, h, y, Y[sample_indices[idx]])
            for idx2 in range(len(grad_W)):
                gW[idx2] += grad_W[idx2]
                gb[idx2] += grad_b[idx2]

            # end of a batch or end of array
            if ((idx + 1)%batch_size == 0) or (idx == len(X) - 1):
                update_W = [momentum*hist_update_W[idx] for idx in range(len(hist_update_W))]
                update_b = [momentum*hist_update_b[idx] for idx in range(len(hist_update_b))]

                for idx2 in range(len(W)):
                    update_W[idx2] += gW[idx2]
                    update_b[idx2] += gb[idx2]
                    W[idx2] -= learning_rate*update_W[idx2]
                    b[idx2] -= learning_rate*update_b[idx2]

                hist_update_W = update_W
                hist_update_b = update_b

                NNmodel.weights = W
                NNmodel.bias = b
                gW = [np.zeros(W[idx].shape) for idx in range(len(W))]
                gb = [np.zeros(b[idx].shape) for idx in range(len(b))]
        
        epoch_acc = NNmodel.evaluate(Xval, Yval)
        print("Epoch : ", iter, "Accuracy : ", epoch_acc*100)
        iter_test_acc.append(epoch_acc)
    
    # plt.figure()
    # plt.plot(iter_test_acc)
    # plt.show()
    
    return NNmodel.weights, NNmodel.bias

def nag(NNmodel, learning_rate, batch_size, epochs, momentum, X, Y, Xval, Yval):
    iter_test_acc = []
    for iter in range(epochs):
        
        W = NNmodel.weights
        b = NNmodel.bias
        gW = [np.zeros(W[idx].shape) for idx in range(len(W))]
        gb = [np.zeros(b[idx].shape) for idx in range(len(b))]

        sample_indices = np.arange(len(X))
        np.random.shuffle(sample_indices)
        hist_update_W = [np.zeros(W[idx].shape) for idx in range(len(W))]
        hist_update_b = [np.zeros(b[idx].shape) for idx in range(len(b))]
        update_W = [np.zeros(W[idx].shape) for idx in range(len(W))]
        update_b = [np.zeros(b[idx].shape) for idx in range(len(b))]

        for idx in range(len(X)):
            a, h, y = NNmodel.forward_propagate(X[sample_indices[idx]])
            grad_W, grad_b = NNmodel.back_propagation(a, h, y, Y[sample_indices[idx]])
            for idx2 in range(len(grad_W)):
                gW[idx2] += grad_W[idx2]
                gb[idx2] += grad_b[idx2]

            # end of a batch or end of array - move by accumulated gradients
            if ((idx + 1)%batch_size == 0) or (idx == len(X) - 1):
                # move by gW as you would already moved by beta*u(t - 1) at the end of the previous batch
                for idx2 in range(len(W)):
                    update_W[idx2] += (gW[idx2])
                    update_b[idx2] += (gb[idx2])
                    W[idx2] -= learning_rate*(gW[idx2])
                    b[idx2] -= learning_rate*(gb[idx2])
                
                hist_update_W = update_W
                hist_update_b = update_b

                NNmodel.weights = W
                NNmodel.bias = b
                gW = [np.zeros(W[idx].shape) for idx in range(len(W))]
                gb = [np.zeros(b[idx].shape) for idx in range(len(b))]
            
            # move by beta*u(t - 1) for next step, however don't do it for the last batch
            if ((idx + 1)%batch_size == 0) and (idx != len(X) - 1):
                update_W = [momentum*hist_update_W[idx] for idx in range(len(hist_update_W))]
                update_b = [momentum*hist_update_b[idx] for idx in range(len(hist_update_b))]
                for idx2 in range(len(W)):
                    W[idx2] -= learning_rate*update_W[idx2]
                    b[idx2] -= learning_rate*update_b[idx2]
        
        epoch_acc = NNmodel.evaluate(Xval, Yval)
        print("Epoch : ", iter, "Accuracy : ", epoch_acc*100)
        iter_test_acc.append(epoch_acc)
    
    # plt.figure()
    # plt.plot(iter_test_acc)
    # plt.show()
    
    return NNmodel.weights, NNmodel.bias

def rmsprop(NNmodel, learning_rate, batch_size, epochs, beta, epsilon, X, Y, Xval, Yval):
    iter_test_acc = []
    for iter in range(epochs):
        
        W = NNmodel.weights
        b = NNmodel.bias
        gW = [np.zeros(W[idx].shape) for idx in range(len(W))]
        gb = [np.zeros(b[idx].shape) for idx in range(len(b))]

        sample_indices = np.arange(len(X))
        np.random.shuffle(sample_indices)
        V_w = [np.zeros(W[idx].shape) for idx in range(len(W))]
        V_b = [np.zeros(b[idx].shape) for idx in range(len(b))]

        for idx in range(len(X)):
            a, h, y = NNmodel.forward_propagate(X[sample_indices[idx]])
            grad_W, grad_b = NNmodel.back_propagation(a, h, y, Y[sample_indices[idx]])
            for idx2 in range(len(grad_W)):
                gW[idx2] += grad_W[idx2]
                gb[idx2] += grad_b[idx2]

            # end of a batch or end of array
            if ((idx + 1)%batch_size == 0) or (idx == len(X) - 1):
                for idx2 in range(len(W)):
                    V_w[idx2] = beta*V_w[idx2] + (1 - beta)*(gW[idx2]**2)
                    V_b[idx2] = beta*V_b[idx2] + (1 - beta)*(gb[idx2]**2)
                    
                    W[idx2] -= learning_rate*gW[idx2]/np.sqrt(V_w[idx2] + epsilon)
                    b[idx2] -= learning_rate*gb[idx2]/np.sqrt(V_b[idx2] + epsilon)

                NNmodel.weights = W
                NNmodel.bias = b
                gW = [np.zeros(W[idx].shape) for idx in range(len(W))]
                gb = [np.zeros(b[idx].shape) for idx in range(len(b))]
        
        epoch_acc = NNmodel.evaluate(Xval, Yval)
        print("Epoch : ", iter, "Accuracy : ", epoch_acc*100)
        iter_test_acc.append(epoch_acc)
    
    # plt.figure()
    # plt.plot(iter_test_acc)
    # plt.show()
    
    return NNmodel.weights, NNmodel.bias
