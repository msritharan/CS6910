"""
This file is used to train a particular model with a specified set of hyperparameters.
The file can be run using the following command structure,
python train.py --dataset [DATASET] ...
To log the data in wandb.ai, also pass "--use_wandb_train True" in the CLI
"""

from get_inputs import *
from neural_network import *
from optimizers import *

import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import pandas as pd
import plotly.express as px
import wandb

# load dataset and process it
if dataset == "fashion_mnist":
    (x_train_original, y_train_original), (x_test_original, y_test_original) = fashion_mnist.load_data()
elif dataset == "mnist":
    (x_train_original, y_train_original), (x_test_original, y_test_original) = mnist.load_data()
x_train = x_train_original.reshape(x_train_original.shape[0], -1)/255
y_train = np.copy(y_train_original)
x_test = x_test_original.reshape(x_test_original.shape[0], -1)/255
y_test = np.copy(y_test_original)

# split train into train + val
train_val_split = 0.9
indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)
total_samples = x_train.shape[0]
Xtrain = np.array([x_train[idx] for idx in range(int(train_val_split*total_samples))])
Ytrain = np.array([y_train[idx] for idx in range(int(train_val_split*total_samples))])
Xval = np.array([x_train[idx] for idx in range(int(train_val_split*total_samples), total_samples)])
Yval = np.array([y_train[idx] for idx in range(int(train_val_split*total_samples), total_samples)])
Xtest = np.array([x_test[idx] for idx in range(len(x_test))])
Ytest = y_test

print("Train Data Dimensions : ", Xtrain.shape)
print("Validation Data Dimensions :", Xval.shape)
print("Test Data Dimensions :", Xtest.shape)

# write training loop for given inputs
model = FeedforwardNN(weight_init, weight_decay, num_layers, hidden_size, activation, input_size, output_size, loss_function)
model.weights, model.bias, train_acc, train_loss, val_acc, val_loss = train_model(model, optimizer, learning_rate, batch_size, epochs, momentum, beta, beta1, beta2, epsilon, Xtrain, Ytrain, Xval, Yval)
    
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
print("Test Accuracy and Loss : ", test_acc, test_loss)

# wandb logs
if use_wandb_train:
    # wandb.login()
    with wandb.init(entity = wandb_entity, project = wandb_project) as run:
        # assign name of the run for easier identification
        name_str = "e_"+ str(epochs) + "_nhl_" + str(num_layers) + "_sz_" + str(hidden_size) + "_w_d_" + str(weight_decay)
        name_str += "_lr_" + str(learning_rate) + "_" + str(optimizer) + "_b_" + str(batch_size)
        name_str += "_" + str(weight_init) + "_" + str(activation) + "_" + dataset + "_" + loss_function
        run.name = name_str

        if dataset == "fashion_mnist":
            class_names = { 0 : "T-shirt",
                            1 : "Trouser",
                            2 : "Pullover",
                            3 : "Dress",
                            4 : "Coat",
                            5 : "Sandal",
                            6 : "Shirt",
                            7 : "Sneaker",
                            8 : "Bag",
                            9 : "Ankle Boot"}
        elif dataset == "mnist":
            class_names = { 0 : "0",
                            1 : "1",
                            2 : "2",
                            3 : "3",
                            4 : "4",
                            5 : "5",
                            6 : "6",
                            7 : "7",
                            8 : "8",
                            9 : "9"}
        
        # log input images
        images = []
        for idx in range(100):
            image = wandb.Image(x_train_original[idx])
            images.append([idx, image, y_train_original[idx], class_names[y_train_original[idx]]])
        columns = ["idx", "image", "label", "class_name"]
        img_table = wandb.Table(data= images, columns= columns)
        wandb.log({ "images_table" : img_table})
        
        # log data for generating confusion matrix
        predictions = []
        true_labels = []
        for idx in range(len(Xtest)):
            predictions.append(model.predict(Xtest[idx]))
            true_labels.append(Ytest[idx])
        conf_mat = np.zeros((len(class_names), len(class_names)))
        for idx in range(len(predictions)):
            conf_mat[true_labels[idx]][len(class_names) - 1 - predictions[idx]] += 1
        conf_mat_df = pd.DataFrame(conf_mat, index = [class_names[i] for i in range(10)], columns = [class_names[len(class_names) - 1 - i] for i in range(10)])
        table = wandb.Table(columns = ["plotly_figure"])
        path_to_plotly_html = "./plotly_figure.html"
        fig = px.imshow(conf_mat_df, color_continuous_scale = "RdYlGn", contrast_rescaling = 'minmax', title= "Confusion Matrix (Predictions vs True Labels)")
        fig.write_html(path_to_plotly_html, auto_play = False)
        table.add_data(wandb.Html(path_to_plotly_html))
        wandb.log({"test_table": table})
        
        # log train and test plots
        epochs = list(range(len(val_acc)))
        wandb.log({"Run Accuracy Plots" : wandb.plot.line_series(xs = epochs, ys = [train_acc, val_acc],
                                        keys = ["train_acc", "val_acc"], title = "Accuracy vs Epoch", xname= "Epochs")})
