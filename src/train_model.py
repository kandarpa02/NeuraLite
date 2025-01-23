import numpy as np
import pandas as pd
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from src.utils import softmax
from src.utils import gradient_descent


data = pd.read_csv("/home/kandarpa-sarkar/Desktop/NeuraLite/data/MNIST_data.csv")

X = data.drop(data.columns[0], axis=1)
y = data[data.columns[0]]

# normalizing the data
X = np.array(X)/ 255.0
# encoding the target value
classes = 10
y = np.eye(classes)[y]

w, b = gradient_descent(X, y, 200, learning_rate=0.01, batch_size=512, tolerance=1e-4, patience=5)

# Save the weights and biases
np.save("/home/kandarpa-sarkar/Desktop/NeuraLite/model/weights.npy", w)
np.save("/home/kandarpa-sarkar/Desktop/NeuraLite/model/biases.npy", b)


model = {
    'weights': w,
    'bias': b
}

# Save the model object to a .pkl file
with open('/home/kandarpa-sarkar/Desktop/NeuraLite/model/trained_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model saved as trained_model.pkl!")

