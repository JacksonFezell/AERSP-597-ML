import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# XNOR function dataset
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[1], [0], [0], [1]])

# Seed for reproducibility
np.random.seed(42)

# Initialize weights randomly with mean 0
input_layer_neurons = inputs.shape[1]
hidden_layer_neurons = 2
output_neuron = 1

# Weights and biases
weights_input_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_layer_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_layer_neurons, output_neuron))
bias_output = np.random.uniform(size=(1, output_neuron))

# Training parameters
learning_rate = 0.1
epochs = 10000

# Training the neural network
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)
    
    # Backpropagation
    error = outputs - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Updating weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += inputs.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# Testing the neural network
print("Final weights from input to hidden layer:\n", weights_input_hidden)
print("Final bias of hidden layer:\n", bias_hidden)
print("Final weights from hidden to output layer:\n", weights_hidden_output)
print("Final bias of output layer:\n", bias_output)

# Predicting output for the XNOR function
hidden_layer_input = np.dot(inputs, weights_input_hidden) + bias_hidden
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
predicted_output = sigmoid(output_layer_input)

print("Predicted output for XNOR function:\n", predicted_output)