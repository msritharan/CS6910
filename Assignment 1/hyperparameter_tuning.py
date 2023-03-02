from neural_network import *
from optimizers import *
from get_inputs import *
import numpy as np
from keras.datasets import fashion_mnist
import wandb
wandb.login()

# Load Dataset
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

# Do Hyperparameter Tuning
sweep_configuration = {
    'name' : 'hyperparameter_tuning',
    'method' : 'random',
    'metric': {'name' : 'val_loss', 'goal' : 'minimize'},
    'parameters' : {
        'epochs': { "values" : [5, 10] },
        'num_hidden_layers': { "values" : [3, 4, 5] },
        'hidden_layer_size': { "values" : [32, 64, 128] },
        'weight_decay': { "values" : [0, 0.0005, 0.5] },
        'learning_rate': { "values" : [1e-3, 1e-4] },
        'optimizer' : { "values" : ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] },
        'batch_size' : { "values" : [16, 32, 64] },
        'weight_init' : { "values" : ["random", "Xavier"] },
        'activation' : { "values" : ["sigmoid", "tanh", "ReLU"] },
        'loss_function' : { "values" : ["mean_squared_error", "cross_entropy"] }
    }
}

sweep_id = wandb.sweep(sweep_configuration, project = wandb_project)

def create_and_train_model(config = None):
    with wandb.init(config = config):
        config = wandb.config
        model = FeedforwardNN(config['weight_init'], config['num_hidden_layers'], config['hidden_layer_size'], config['activation'],
                            input_size, output_size, config['loss_function'])
        model.weights, model.bias, train_acc, train_loss, val_acc, val_loss = train_model(model, config['optimizer'], config['learning_rate'], config['batch_size'],
                                                                                config['epochs'], momentum, beta, beta1, beta2, epsilon, Xtrain, Ytrain, Xval, Yval)
        for epoch in range(config['epochs']):
            wandb.log({
                'train_acc': train_acc[epoch],
                'train_loss': train_loss[epoch],
                'val_acc': val_acc[epoch],
                'val_loss': val_loss[epoch],
                'epoch' : epoch
            })


agent = wandb.agent(sweep_id, function = create_and_train_model, project = wandb_project) 
