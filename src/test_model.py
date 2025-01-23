import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import accuracy


data = pd.read_csv('data/MNIST_data.csv')

X = data.drop(data.columns[0], axis=1)
y = data[data.columns[0]]
classes = 10
y = np.eye(classes)[y] # encoding data

# Accessing 'w' and 'b' from the loaded file
w = np.load('model/weights.npy')
b = np.load('model/biases.npy')

print(accuracy(X, y, w, b))