
# CS6910: Deep Learning - Assignment 3

The assignment focuses on using recurrent neural networks to build a transliteration system. The assignment was done by Manikandan Sritharan, EE19B038.

Link to the WandB report: https://wandb.ai/mani-ml/CS6910-A3/reports/CS6910-Assignment-3-Report--Vmlldzo0NDIyNzU2

## Dataset

The models trained and reported were based on the Tamil dataset given in the project statement.

Upload the dataset to Google Drive before running the notebooks.
## Running the Code

The code is structured into two Colab Notebooks. To run the notebooks and document the results, you can change the sweep_config to the desired state and run it. (Do ensure to change the number of runs in wandb agent as per your preference)

To run the code without WandB, uncomment the commented out part before the "Wandb Sweeps" section of the notebooks and run it.
## Predictions on Test Data

The predictions of the models on the test data has been uploaded to the predictions_vanilla and predictions_attention folders respectively