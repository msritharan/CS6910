import numpy as np
import pandas as pd

class FeedforwardNN:
    def __init__(self, layers):
        # contains sizes of input layer, h1, h2, ..., output layer
        self.layers = layers 
        self.weights
