import numpy as np

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) 
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(Prob, y):
    N = y.shape[0]
    loss = -np.sum(y * np.log(Prob + 1e-15)) / N
    return loss

import numpy as np

import numpy as np

def gradient_descent(X, y, epochs, learning_rate, batch_size, tolerance=1e-4, patience=5):
    """
    Gradient descent optimization without L2 regularization and with early stopping.
    
    Args:
        X (numpy.ndarray): Input data of shape (num_samples, num_features)
        y (numpy.ndarray): Target labels, one-hot encoded of shape (num_samples, num_classes)
        epochs (int): Number of epochs to train the model
        learning_rate (float): Learning rate for gradient descent
        batch_size (int): Size of each mini-batch
        tolerance (float): Minimum change in loss to trigger early stopping
        patience (int): Number of epochs to wait for improvement before stopping early
    
    Returns:
        numpy.ndarray: Optimized weights (W)
        numpy.ndarray: Optimized biases (b)
    """
    num_samples, num_features = X.shape
    num_classes = y.shape[1]
    
    # Initialize weights and biases
    W = np.zeros((num_features, num_classes))  # Shape: (features x classes)
    b = np.zeros((1, num_classes))              # Shape: (1 x classes)

    # Track the previous loss for early stopping
    previous_loss = float('inf')
    patience_counter = 0

    # Iterate over epochs
    for epoch in range(epochs):
        # Shuffle the dataset at the beginning of each epoch
        indices = np.random.permutation(num_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Process in mini-batches
        for batch_start in range(0, num_samples, batch_size):
            # Get the current mini-batch
            X_batch = X_shuffled[batch_start:batch_start + batch_size]
            y_batch = y_shuffled[batch_start:batch_start + batch_size]

            # Forward pass
            Z = np.dot(X_batch, W) + b  # Shape: (batch_size x num_classes)
            Prob = softmax(Z)           # Shape: (batch_size x num_classes)

            # Compute loss (no regularization)
            loss = cross_entropy_loss(Prob, y_batch)

            # Compute gradients
            dW = np.dot(X_batch.T, (Prob - y_batch)) / batch_size  # Gradient for weights
            db = np.sum(Prob - y_batch, axis=0, keepdims=True) / batch_size  # Gradient for biases

            # Update parameters
            W -= learning_rate * dW
            b -= learning_rate * db

        # Early stopping condition
        if abs(previous_loss - loss) < tolerance:
            print(f"Early stopping at epoch {epoch}, loss change is below tolerance.")
            break
        
        # If loss doesn't improve, increase the patience counter
        if loss > previous_loss:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}, patience exceeded.")
                break
        else:
            patience_counter = 0
        
        # Update the previous loss for the next iteration
        previous_loss = loss
        
        # Print loss every 10 epochs (or the last epoch)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return W, b  # Return the best weights and biases that minimize loss




def predict(X, W, b):

    Z = np.dot(X, W) + b               # Compute logits
    Prob = softmax(Z)                  # Apply softmax to get probabilities
    predictions = np.argmax(Prob, axis=1)  # Get the class with the highest probability
    return predictions

def accuracy(X, y, W, b):
    y_pred = predict(X, W, b)
    accuracy = np.mean(y_pred == np.argmax(y, axis=1))
    return f"accuracy: {accuracy*100}%"