'''
get_inputs.py 
initiates and assigns default values to parameters and processes command line arguments
continues with default values for input errors
'''

import sys

# Default Arguments
wandb_project = "CS6910 Assignment 1"
wandb_entity = "manikandan_sritharan"
dataset = "fashion_mnist"
epochs = 1
batch_size = 4
loss_function = "cross_entropy"
optimizer = "sgd"
learning_rate = 0.1
momentum = 0.5
beta = 0.5
beta1 = 0.5
beta2 = 0.5
epsilon = 0.000001
weight_decay = 0.0
weight_init = "random"
num_layers = 1
hidden_size = 4
activation = "sigmoid"

# Input Arguments
pos = 2
while(pos < len(sys.argv)):

    variable = sys.argv[pos - 1]
    value = sys.argv[pos]
    print(pos, variable, value)
    
    if variable in ["-wp", "--wandb_project"]:
        wandb_project = value

    elif variable in ["-we", "--wand_entity"]:
        wandb_entity = value

    elif variable in ["-d", "--dataset"]:
        if value in ["mnist", "fashion_mnist"]:
            dataset = value
        else:
            print("Invalid Choice for dataset.")
            print("Assigning Default Value = fashion_mnist")
    
    elif variable in ["-e", "--epochs"]:
        try:
            epochs = int(value)
        except ValueError:
            print("Not a valid number for epochs")
            print("Proceeding with Default Value.")
    
    elif variable in ["-b", "--batch_size"]:
        try:
            batch_size = int(value)
        except ValueError:
            print("Not a valid number for batch_size")
            print("Proceeding with Default Value.")
    
    elif variable in ["-b", "--loss"]:
        if value in ["mean_squared_error", "cross_entropy"]:
            loss_function = value
        else:
            print("Invalid choice for Loss Function.")
            print("Assigning Default Value = cross_entropy")
            
    elif variable in ["-o", "--optimizer"]:
        if value in ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]:
            optimizer = value
        else:
            print("Invalid Choice for optimizer.")
            print("Assigning Default Value = sgd")
            
    elif variable in ["-lr", "--learning_rate"]:
        try:
            learning_rate = float(value)
        except ValueError:
            print("Not a valid float for learning_rate")
            print("Proceeding with Default Value.")
    
    elif variable in ["-m", "--momentum"]:
        try:
            momentum = float(value)
        except ValueError:
            print("Not a valid float for momentum")
            print("Proceeding with Default Value.")
    
    elif variable in ["-beta", "--beta"]:
        try:
            beta = float(value)
        except ValueError:
            print("Not a valid float for beta")
            print("Proceeding with Default Value.")
    
    elif variable in ["-beta1", "--beta1"]:
        try:
            beta1 = float(value)
        except ValueError:
            print("Not a valid float for beta1")
            print("Proceeding with Default Value.")

    elif variable in ["-beta2", "--beta2"]:
        try:
            beta2 = float(value)
        except ValueError:
            print("Not a valid float for beta2")
            print("Proceeding with Default Value.")
    
    elif variable in ["-eps", "--epsilon"]:
        try:
            epsilon = float(value)
        except ValueError:
            print("Not a valid float for epsilon")
            print("Proceeding with Default Value.")

    elif variable in ["-w_d", "--weight_decay"]:
        try:
            weight_decay = float(value)
        except ValueError:
            print("Not a valid float for weight_decay")
            print("Proceeding with Default Value.")

    elif variable in ["-w_i", "--weight_init"]:
        if value in ["random", "Xavier"]:
            weight_init = value
        else:
            print("Invalid Choice for weight_init.")
            print("Assigning Default Value = random")
    
    elif variable in ["-nhl", "--num_layers"]:
        try:
            num_layers = int(value)
        except ValueError:
            print("Not a valid int for num_layers")
            print("Proceeding with Default Value.")
    
    elif variable in ["-sz", "--hidden_size"]:
        try:
            hidden_size = int(value)
        except ValueError:
            print("Not a valid int for hidden_size")
            print("Proceeding with Default Value.")

    elif variable in ["-a", "--activation"]:
        if value in ["identity", "sigmoid", "tanh", "ReLU"]:
            activation = value
        else:
            print("Invalid Choice for activation.")
            print("Assigning Default Value = sigmoid")
    
    else:
        print("Unknown argument :", variable)
        print("Ignoring argument.")
    
    pos += 2