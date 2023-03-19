"""
This file contains the code for the following optimizer functions:
- Mini-batch Stochastic Gradient Descent (sgd)
- Momentum based Mini Batch SGD (momentum_sgd)
- Nesterov Accelerated Mini Batch SGD (nag)
- RMSprop (rmsprop)
- Adam (adam)
- Nadam (nadam)

The file also contains the function train_model() which trains a model using the specified input parameters.
"""

import numpy as np

def sgd(NNmodel, learning_rate, batch_size, epochs, Xtrain, Ytrain, Xval, Yval):
    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []

    for iter in range(epochs):
        
        W = NNmodel.weights
        b = NNmodel.bias
        gW = [np.zeros(W[idx].shape) for idx in range(len(W))]
        gb = [np.zeros(b[idx].shape) for idx in range(len(b))]

        sample_indices = np.arange(len(Xtrain))
        np.random.shuffle(sample_indices)

        for idx in range(len(Xtrain)):
            a, h, y = NNmodel.forward_propagate(Xtrain[sample_indices[idx]])
            grad_W, grad_b = NNmodel.back_propagation(a, h, y, Ytrain[sample_indices[idx]])
            for idx2 in range(len(grad_W)):
                gW[idx2] += grad_W[idx2]
                gb[idx2] += grad_b[idx2]

            # end of a batch or end of array
            if ((idx + 1)%batch_size == 0) or (idx == len(Xtrain) - 1):
                g_weight_loss, g_bias_loss = NNmodel.grad_weight_loss()
                for idx2 in range(len(W)):
                    W[idx2] -= learning_rate*(gW[idx2] + g_weight_loss[idx2])
                    b[idx2] -= learning_rate*(gb[idx2] + g_bias_loss[idx2])

                NNmodel.weights = W
                NNmodel.bias = b
                gW = [np.zeros(W[idx].shape) for idx in range(len(W))]
                gb = [np.zeros(b[idx].shape) for idx in range(len(b))]
        
        train_acc, train_cost = NNmodel.evaluate_metrics(Xtrain, Ytrain)
        val_acc, val_cost = NNmodel.evaluate_metrics(Xval, Yval)
    
        print("Epoch : ", iter + 1, "Train Accuracy = ", train_acc, "Val Accuracy = ", val_acc)
        train_accuracy.append(train_acc)
        train_loss.append(train_cost) 
        val_accuracy.append(val_acc)
        val_loss.append(val_cost)

    return NNmodel.weights, NNmodel.bias, train_accuracy, train_loss, val_accuracy, val_loss

def momentum_sgd(NNmodel, learning_rate, batch_size, epochs, momentum, Xtrain, Ytrain, Xval, Yval):
    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []

    for iter in range(epochs):
        
        W = NNmodel.weights
        b = NNmodel.bias
        gW = [np.zeros(W[idx].shape) for idx in range(len(W))]
        gb = [np.zeros(b[idx].shape) for idx in range(len(b))]

        sample_indices = np.arange(len(Xtrain))
        np.random.shuffle(sample_indices)
        hist_update_W = [np.zeros(W[idx].shape) for idx in range(len(W))]
        hist_update_b = [np.zeros(b[idx].shape) for idx in range(len(b))]

        for idx in range(len(Xtrain)):
            a, h, y = NNmodel.forward_propagate(Xtrain[sample_indices[idx]])
            grad_W, grad_b = NNmodel.back_propagation(a, h, y, Ytrain[sample_indices[idx]])
            for idx2 in range(len(grad_W)):
                gW[idx2] += grad_W[idx2]
                gb[idx2] += grad_b[idx2]

            # end of a batch or end of array
            if ((idx + 1)%batch_size == 0) or (idx == len(Xtrain) - 1):
                update_W = [momentum*hist_update_W[idx] for idx in range(len(hist_update_W))]
                update_b = [momentum*hist_update_b[idx] for idx in range(len(hist_update_b))]
                g_weight_loss, g_bias_loss = NNmodel.grad_weight_loss()
                for idx2 in range(len(W)):
                    update_W[idx2] += (gW[idx2] + g_weight_loss[idx2])
                    update_b[idx2] += (gb[idx2] + g_bias_loss[idx2])
                    W[idx2] -= learning_rate*update_W[idx2]
                    b[idx2] -= learning_rate*update_b[idx2]

                hist_update_W = update_W
                hist_update_b = update_b

                NNmodel.weights = W
                NNmodel.bias = b
                gW = [np.zeros(W[idx].shape) for idx in range(len(W))]
                gb = [np.zeros(b[idx].shape) for idx in range(len(b))]
        
        train_acc, train_cost = NNmodel.evaluate_metrics(Xtrain, Ytrain)
        val_acc, val_cost = NNmodel.evaluate_metrics(Xval, Yval)
        print("Epoch : ", iter + 1, "Train Accuracy = ", train_acc, "Val Accuracy = ", val_acc)
        train_accuracy.append(train_acc)
        train_loss.append(train_cost) 
        val_accuracy.append(val_acc)
        val_loss.append(val_cost)
    
    return NNmodel.weights, NNmodel.bias, train_accuracy, train_loss, val_accuracy, val_loss

def nag(NNmodel, learning_rate, batch_size, epochs, momentum, Xtrain, Ytrain, Xval, Yval):
    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []

    for iter in range(epochs):
        
        W = NNmodel.weights
        b = NNmodel.bias
        gW = [np.zeros(W[idx].shape) for idx in range(len(W))]
        gb = [np.zeros(b[idx].shape) for idx in range(len(b))]

        sample_indices = np.arange(len(Xtrain))
        np.random.shuffle(sample_indices)
        hist_update_W = [np.zeros(W[idx].shape) for idx in range(len(W))]
        hist_update_b = [np.zeros(b[idx].shape) for idx in range(len(b))]
        update_W = [np.zeros(W[idx].shape) for idx in range(len(W))]
        update_b = [np.zeros(b[idx].shape) for idx in range(len(b))]

        for idx in range(len(Xtrain)):
            a, h, y = NNmodel.forward_propagate(Xtrain[sample_indices[idx]])
            grad_W, grad_b = NNmodel.back_propagation(a, h, y, Ytrain[sample_indices[idx]])
            for idx2 in range(len(grad_W)):
                gW[idx2] += grad_W[idx2]
                gb[idx2] += grad_b[idx2]

            # end of a batch or end of array - move by accumulated gradients
            if ((idx + 1)%batch_size == 0) or (idx == len(Xtrain) - 1):
                # move by gW as you would already moved by beta*u(t - 1) at the end of the previous batch
                g_weight_loss, g_bias_loss = NNmodel.grad_weight_loss()
                for idx2 in range(len(W)):
                    update_W[idx2] += (gW[idx2] + g_weight_loss[idx2])
                    update_b[idx2] += (gb[idx2] + g_bias_loss[idx2])
                    W[idx2] -= learning_rate*(gW[idx2])
                    b[idx2] -= learning_rate*(gb[idx2])
                
                hist_update_W = update_W
                hist_update_b = update_b

                NNmodel.weights = W
                NNmodel.bias = b
                gW = [np.zeros(W[idx].shape) for idx in range(len(W))]
                gb = [np.zeros(b[idx].shape) for idx in range(len(b))]
            
            # move by beta*u(t - 1) for next step, however don't do it for the last batch
            if ((idx + 1)%batch_size == 0) and (idx != len(Xtrain) - 1):
                update_W = [momentum*hist_update_W[idx] for idx in range(len(hist_update_W))]
                update_b = [momentum*hist_update_b[idx] for idx in range(len(hist_update_b))]
                for idx2 in range(len(W)):
                    W[idx2] -= learning_rate*update_W[idx2]
                    b[idx2] -= learning_rate*update_b[idx2]
        
        train_acc, train_cost = NNmodel.evaluate_metrics(Xtrain, Ytrain)
        val_acc, val_cost = NNmodel.evaluate_metrics(Xval, Yval)
        print("Epoch : ", iter + 1, "Train Accuracy = ", train_acc, "Val Accuracy = ", val_acc)
        train_accuracy.append(train_acc)
        train_loss.append(train_cost) 
        val_accuracy.append(val_acc)
        val_loss.append(val_cost)
    
    return NNmodel.weights, NNmodel.bias, train_accuracy, train_loss, val_accuracy, val_loss

def rmsprop(NNmodel, learning_rate, batch_size, epochs, beta, epsilon, Xtrain, Ytrain, Xval, Yval):
    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []

    for iter in range(epochs):
        
        W = NNmodel.weights
        b = NNmodel.bias
        gW = [np.zeros(W[idx].shape) for idx in range(len(W))]
        gb = [np.zeros(b[idx].shape) for idx in range(len(b))]

        sample_indices = np.arange(len(Xtrain))
        np.random.shuffle(sample_indices)
        V_w = [np.zeros(W[idx].shape) for idx in range(len(W))]
        V_b = [np.zeros(b[idx].shape) for idx in range(len(b))]

        for idx in range(len(Xtrain)):
            a, h, y = NNmodel.forward_propagate(Xtrain[sample_indices[idx]])
            grad_W, grad_b = NNmodel.back_propagation(a, h, y, Ytrain[sample_indices[idx]])
            for idx2 in range(len(grad_W)):
                gW[idx2] += grad_W[idx2]
                gb[idx2] += grad_b[idx2]

            # end of a batch or end of array
            if ((idx + 1)%batch_size == 0) or (idx == len(Xtrain) - 1):
                g_weight_loss, g_bias_loss = NNmodel.grad_weight_loss()
                for idx2 in range(len(W)):
                    V_w[idx2] = beta*V_w[idx2] + (1 - beta)*((gW[idx2] + g_weight_loss[idx2])**2)
                    V_b[idx2] = beta*V_b[idx2] + (1 - beta)*((gb[idx2] + g_bias_loss[idx2])**2)
                    
                    W[idx2] -= learning_rate*(gW[idx2] + g_weight_loss[idx2])/np.sqrt(V_w[idx2] + epsilon)
                    b[idx2] -= learning_rate*(gb[idx2] + g_bias_loss[idx2])/np.sqrt(V_b[idx2] + epsilon)

                NNmodel.weights = W
                NNmodel.bias = b
                gW = [np.zeros(W[idx].shape) for idx in range(len(W))]
                gb = [np.zeros(b[idx].shape) for idx in range(len(b))]
        
        train_acc, train_cost = NNmodel.evaluate_metrics(Xtrain, Ytrain)
        val_acc, val_cost = NNmodel.evaluate_metrics(Xval, Yval)
        print("Epoch : ", iter + 1, "Train Accuracy = ", train_acc, "Val Accuracy = ", val_acc)
        train_accuracy.append(train_acc)
        train_loss.append(train_cost) 
        val_accuracy.append(val_acc)
        val_loss.append(val_cost)
    
    return NNmodel.weights, NNmodel.bias, train_accuracy, train_loss, val_accuracy, val_loss

def adam(NNmodel, learning_rate, batch_size, epochs, beta1, beta2, epsilon, Xtrain, Ytrain, Xval, Yval):
    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []

    for iter in range(epochs):
        
        W = NNmodel.weights
        b = NNmodel.bias
        gW = [np.zeros(W[idx].shape) for idx in range(len(W))]
        gb = [np.zeros(b[idx].shape) for idx in range(len(b))]

        sample_indices = np.arange(len(Xtrain))
        np.random.shuffle(sample_indices)
        M_w = [np.zeros(W[idx].shape) for idx in range(len(W))]
        M_b = [np.zeros(b[idx].shape) for idx in range(len(b))]
        V_w = [np.zeros(W[idx].shape) for idx in range(len(W))]
        V_b = [np.zeros(b[idx].shape) for idx in range(len(b))]

        t = 1
        for idx in range(len(Xtrain)):
            a, h, y = NNmodel.forward_propagate(Xtrain[sample_indices[idx]])
            grad_W, grad_b = NNmodel.back_propagation(a, h, y, Ytrain[sample_indices[idx]])
            for idx2 in range(len(grad_W)):
                gW[idx2] += grad_W[idx2]
                gb[idx2] += grad_b[idx2]

            # end of a batch or end of array
            if ((idx + 1)%batch_size == 0) or (idx == len(Xtrain) - 1):
                g_weight_loss, g_bias_loss = NNmodel.grad_weight_loss()
                for idx2 in range(len(W)):
                    M_w[idx2] = beta1*M_w[idx2] + (1 - beta1)*((gW[idx2] + g_weight_loss[idx2]))
                    M_b[idx2] = beta1*M_b[idx2] + (1 - beta1)*((gb[idx2] + g_bias_loss[idx2]))
                    V_w[idx2] = beta2*V_w[idx2] + (1 - beta2)*((gW[idx2] + g_weight_loss[idx2])**2)
                    V_b[idx2] = beta2*V_b[idx2] + (1 - beta2)*((gb[idx2] + g_bias_loss[idx2])**2)
                    
                    normal_M_w = M_w[idx2]/(1 - pow(beta1, t))
                    normal_M_b = M_b[idx2]/(1 - pow(beta1, t))
                    normal_V_w = V_w[idx2]/(1 - pow(beta2, t))
                    normal_V_b = V_b[idx2]/(1 - pow(beta2, t))

                    W[idx2] -= learning_rate*normal_M_w/(np.sqrt(normal_V_w) + epsilon)
                    b[idx2] -= learning_rate*normal_M_b/(np.sqrt(normal_V_b) + epsilon)
                
                t += 1
                NNmodel.weights = W
                NNmodel.bias = b
                gW = [np.zeros(W[idx].shape) for idx in range(len(W))]
                gb = [np.zeros(b[idx].shape) for idx in range(len(b))]
        
        train_acc, train_cost = NNmodel.evaluate_metrics(Xtrain, Ytrain)
        val_acc, val_cost = NNmodel.evaluate_metrics(Xval, Yval)
        print("Epoch : ", iter + 1, "Train Accuracy = ", train_acc, "Val Accuracy = ", val_acc)
        train_accuracy.append(train_acc)
        train_loss.append(train_cost) 
        val_accuracy.append(val_acc)
        val_loss.append(val_cost)
    
    return NNmodel.weights, NNmodel.bias, train_accuracy, train_loss, val_accuracy, val_loss

def nadam(NNmodel, learning_rate, batch_size, epochs, beta1, beta2, epsilon, Xtrain, Ytrain, Xval, Yval):
    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []

    for iter in range(epochs):
        
        W = NNmodel.weights
        b = NNmodel.bias
        gW = [np.zeros(W[idx].shape) for idx in range(len(W))]
        gb = [np.zeros(b[idx].shape) for idx in range(len(b))]

        sample_indices = np.arange(len(Xtrain))
        np.random.shuffle(sample_indices)
        M_w = [np.zeros(W[idx].shape) for idx in range(len(W))]
        M_b = [np.zeros(b[idx].shape) for idx in range(len(b))]
        V_w = [np.zeros(W[idx].shape) for idx in range(len(W))]
        V_b = [np.zeros(b[idx].shape) for idx in range(len(b))]

        t = 1
        for idx in range(len(Xtrain)):
            a, h, y = NNmodel.forward_propagate(Xtrain[sample_indices[idx]])
            grad_W, grad_b = NNmodel.back_propagation(a, h, y, Ytrain[sample_indices[idx]])
            for idx2 in range(len(grad_W)):
                gW[idx2] += grad_W[idx2]
                gb[idx2] += grad_b[idx2]

            # end of a batch or end of array
            if ((idx + 1)%batch_size == 0) or (idx == len(Xtrain) - 1):
                g_weight_loss, g_bias_loss = NNmodel.grad_weight_loss()
                for idx2 in range(len(W)):
                    M_w[idx2] = beta1*M_w[idx2] + (1 - beta1)*((gW[idx2] + g_weight_loss[idx2]))
                    M_b[idx2] = beta1*M_b[idx2] + (1 - beta1)*((gb[idx2] + g_bias_loss[idx2]))
                    V_w[idx2] = beta2*V_w[idx2] + (1 - beta2)*((gW[idx2] + g_weight_loss[idx2])**2)
                    V_b[idx2] = beta2*V_b[idx2] + (1 - beta2)*((gb[idx2] + g_bias_loss[idx2])**2)
                    
                    normal_M_w = M_w[idx2]/(1 - pow(beta1, t))
                    normal_M_b = M_b[idx2]/(1 - pow(beta1, t))
                    normal_V_w = V_w[idx2]/(1 - pow(beta2, t))
                    normal_V_b = V_b[idx2]/(1 - pow(beta2, t))

                    W[idx2] -= learning_rate*(beta1*normal_M_w + (1 - beta1)*(gW[idx2] + g_weight_loss[idx2])/(1 - pow(beta1, t)))/(np.sqrt(normal_V_w) + epsilon)
                    b[idx2] -= learning_rate*(beta1*normal_M_b + (1 - beta1)*(gb[idx2] + g_bias_loss[idx2])/(1 - pow(beta1, t)))/(np.sqrt(normal_V_b) + epsilon)

                t += 1
                NNmodel.weights = W
                NNmodel.bias = b
                gW = [np.zeros(W[idx].shape) for idx in range(len(W))]
                gb = [np.zeros(b[idx].shape) for idx in range(len(b))]
        
        train_acc, train_cost = NNmodel.evaluate_metrics(Xtrain, Ytrain)
        val_acc, val_cost = NNmodel.evaluate_metrics(Xval, Yval)
        print("Epoch : ", iter + 1, "Train Accuracy = ", train_acc, "Val Accuracy = ", val_acc)
        train_accuracy.append(train_acc)
        train_loss.append(train_cost) 
        val_accuracy.append(val_acc)
        val_loss.append(val_cost)
    
    return NNmodel.weights, NNmodel.bias, train_accuracy, train_loss, val_accuracy, val_loss

def train_model(NNmodel, optimizer, learning_rate, batch_size, epochs, momentum, beta, beta1, beta2, epsilon, Xtrain, Ytrain, Xval, Yval):
    # train_accuracy, train_loss, val_accuracy, val_loss
    if optimizer == "sgd":
        W, b, train_acc, train_loss, val_acc, val_loss = sgd(NNmodel, learning_rate, batch_size, epochs, Xtrain, Ytrain, Xval, Yval)
    elif optimizer == "momentum":
        W, b, train_acc, train_loss, val_acc, val_loss = momentum_sgd(NNmodel, learning_rate, batch_size, epochs, momentum, Xtrain, Ytrain, Xval, Yval)
    elif optimizer == "nag":
        W, b, train_acc, train_loss, val_acc, val_loss = nag(NNmodel, learning_rate, batch_size, epochs, momentum, Xtrain, Ytrain, Xval, Yval)
    elif optimizer == "rmsprop":
        W, b, train_acc, train_loss, val_acc, val_loss = rmsprop(NNmodel, learning_rate, batch_size, epochs, beta, epsilon, Xtrain, Ytrain, Xval, Yval)
    elif optimizer == "adam":
        W, b, train_acc, train_loss, val_acc, val_loss = adam(NNmodel, learning_rate, batch_size, epochs, beta1, beta2, epsilon, Xtrain, Ytrain, Xval, Yval)
    elif optimizer == "nadam":
        W, b, train_acc, train_loss, val_acc, val_loss = nadam(NNmodel, learning_rate, batch_size, epochs, beta1, beta2, epsilon, Xtrain, Ytrain, Xval, Yval)

    NNmodel.weights = W
    NNmodel.bias = b

    return NNmodel.weights, NNmodel.bias, train_acc, train_loss, val_acc, val_loss
