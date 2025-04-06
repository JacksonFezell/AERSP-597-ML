import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
*******************************
*                             *
*    Problem 1 Definitions    *
*                             *
*******************************
"""

# Sigmoid activation function
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output

# Derivative of the sigmoid function, 
def sigmoid_deriv(y):
    return y * (1 - y)

def q1a_main():
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
        d_predicted_output = error * sigmoid_deriv(predicted_output)
        
        error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_deriv(hidden_layer_output)
        
        # Updating weights and biases
        weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
        bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
        weights_input_hidden += inputs.T.dot(d_hidden_layer) * learning_rate
        bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    print("Final weights from input to hidden layer:\n", weights_input_hidden)
    print("Final bias of hidden layer:\n", bias_hidden)
    print("Final weights from hidden to output layer:\n", weights_hidden_output)
    print("Final bias of output layer:\n", bias_output)

    # Predicting output for the XNOR function
    hidden_layer_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    print("Inputs to the XNOR function:\n", inputs)
    print("Predicted output:\n", np.round(predicted_output))
    print("Prediction percentage for XNOR function:\n", predicted_output)


"""
*******************************
*                             *
*    Problem 2 Definitions    *
*                             *
*******************************
"""

def swish(x):
    output = x / (1 + np.exp(-x))
    #print(f"Swish output: {output}")
    return output
    

def relu(x):
    return np.maximum(0, x)

def relu_deriv(y):
    return np.where(y > 0, 1, 0)

def forwardprop(x, t, A, B, C):
    x_bias = np.vstack([x, np.ones((1, x.shape[1]))]) 
    y = swish(A @ x_bias)

    y_bias = np.vstack([y, np.ones((1, y.shape[1]))])
    z = swish(B @ y_bias)

    z_bias = np.vstack([z, np.ones((1, z.shape[1]))])
    h = relu(C @ z_bias)

    E = 0.5 * np.mean((h - t) ** 2)  # Mean Squared Error
    #print(f"MSE: {E}")
    E_grad = 2 * (h - t) / h.shape[1]
    #print(f"Gradient of MSE: {E_grad}")
    return y, z, h, E

def backprop(x, t, A, B, C):
    y, z, h, E = forwardprop(x, t, A, B, C)
    
    # Output layer weights
    del_C = (h - t) * relu_deriv(h)
    grad_C = del_C @ np.vstack([z, np.ones((1, z.shape[1]))]).T

    # Second hidden layer weights
    del_B = (C[:, :-1].T @ del_C) * swish_deriv(z) 
    grad_B = del_B @ np.vstack([y, np.ones((1, y.shape[1]))]).T 

    # First hidden layer weights
    del_A = (B[:, :-1].T @ del_B) * swish_deriv(y) 
    grad_A = del_A @ np.vstack([x, np.ones((1, x.shape[1]))]).T 
    
    return grad_A, grad_B, grad_C

def gradient_check():

    epsilon = 1e-4
    M = 100
    K = 50
    D = 30
    N = 10
    iter = 1000

    A = np.random.rand(K, M + 1) * 0.1 - 0.05
    B = np.random.rand(D, K + 1) * 0.1 - 0.05
    C = np.random.rand(N, D + 1) * 0.1 - 0.05
    x = np.random.rand(M, 1) * 0.1 - 0.05
    t = np.random.rand(N, 1) * 0.2 - 0.1

    grad_A, grad_B, grad_C = backprop(x, t, A, B, C)
    errA, errB, errC = [], [], []

    for i in range(iter):
        idx_x = np.random.randint(0, A.shape[0])
        idx_y = np.random.randint(0, A.shape[1])
        
        A[idx_x, idx_y] += epsilon
        _, _, _, E_plus = forwardprop(x, t, A, B, C)
        A[idx_x, idx_y] -= 2 * epsilon
        _, _, _, E_minus = forwardprop(x, t, A, B, C)
        A[idx_x, idx_y] += epsilon

        # Central difference
        numerical_grad_A = (E_plus - E_minus) / (2 * epsilon)
        errA.append(np.abs(grad_A[idx_x, idx_y] - numerical_grad_A))

        idx_x = np.random.randint(0, B.shape[0])
        idx_y = np.random.randint(0, B.shape[1])

        B[idx_x, idx_y] += epsilon
        _, _, _, E_plus = forwardprop(x, t, A, B, C)
        B[idx_x, idx_y] -= 2 * epsilon
        _, _, _, E_minus = forwardprop(x, t, A, B, C)
        B[idx_x, idx_y] += epsilon

        # Central difference
        numerical_grad_B = (E_plus - E_minus) / (2 * epsilon)
        errB.append(np.abs(grad_B[idx_x, idx_y] - numerical_grad_B))
        
        idx_x = np.random.randint(0, C.shape[0])
        idx_y = np.random.randint(0, C.shape[1])

        C[idx_x, idx_y] += epsilon
        _, _, _, E_plus = forwardprop(x, t, A, B, C)
        C[idx_x, idx_y] -= 2 * epsilon
        _, _, _, E_minus = forwardprop(x, t, A, B, C)
        C[idx_x, idx_y] += epsilon

        # Central difference
        numerical_grad_C = (E_plus - E_minus) / (2 * epsilon)
        errC.append(np.abs(grad_C[idx_x, idx_y] - numerical_grad_C))

    print('Gradient checking A, MAE: {0:0.8f}'.format(np.mean(errA)))
    print('Gradient checking B, MAE: {0:0.8f}'.format(np.mean(errB)))
    print('Gradient checking C, MAE: {0:0.8f}'.format(np.mean(errC)))

    # Plot the errors over nCheck
    plt.figure(figsize=(10, 6))
    plt.plot(range(iter), errA, label='Error A', color='blue')
    plt.plot(range(iter), errB, label='Error B', color='green')
    plt.plot(range(iter), errC, label='Error C', color='red')
    plt.xlabel('Gradient Check Iterations')
    plt.ylabel('Error')
    plt.title('Gradient Check Errors')
    plt.legend()
    plt.grid(True)
    plt.show()


"""
*******************************
*                             *
*    Problem 3 Definitions    *
*                             *
*******************************
"""

def min_max(data):
    """
    Min-max normalize the data to the specified feature range.
    """
    min_val = np.min(data, axis=1, keepdims=True)
    max_val = np.max(data, axis=1, keepdims=True)

    diff_val = max_val - min_val
    diff_val[diff_val == 0] = 1e-8

    normalized_data = (data - min_val) / (diff_val)

    return normalized_data

def weight_init(input_size, output_size, activation, distrib):
    fan_in = input_size
    fan_out = output_size
    n = (fan_in + fan_out) / 2

    # He initialization for ReLU layer activation
    if activation == 'relu': 
        if distrib == 'uniform':
            limit = np.sqrt(6 / fan_in)
            return np.random.uniform(-limit, limit, (output_size, input_size + 1))
        else:  # gaussian
            std = np.sqrt(2 / fan_in)
            return np.random.normal(0, std, (output_size, input_size + 1))
        
    # Xavier for sigmoid/tanh layer activation
    else:  
        if distrib == 'uniform':
            limit = np.sqrt(3 / n)
            return np.random.uniform(-limit, limit, (output_size, input_size + 1))
        else:  # gaussian
            std = np.sqrt(1 / n)
            return np.random.normal(0, std, (output_size, input_size + 1))


def swish_deriv(a):
    sigma = 1 / (1 + np.exp(-a))
    return sigma + a * sigma * (1 - sigma)

def SGD_training(data, target, A, B, C, learning_rate, epochs):

    best_A, best_B, best_C = A.copy(), B.copy(), C.copy()
    best_loss = float('inf')
    train_losses = []
    test_losses = []

    # Training loop
    for epoch in range(epochs):
        # Perform SGD
        # for i in range(data.shape[1]):
            # x_sample = data[:, i:i+1]
            # t_sample = target[:, i:i+1]

        # Forward and backward propagation
        grad_A, grad_B, grad_C = backprop(data, target, A, B, C)
        print("Gradients A:", grad_A)
        print("Gradients B:", grad_B)
        print("Gradients C:", grad_C)


        # SGD fixed learning rate
        A -= learning_rate * grad_A
        B -= learning_rate * grad_B
        C -= learning_rate * grad_C

        # Compute training loss for the epoch
        _, _, _, train_loss = forwardprop(data, target, A, B, C)
        train_losses.append(train_loss)

        # Compute test loss for the epoch
        _, _, _, test_loss = forwardprop(data, target, A, B, C)
        test_losses.append(test_loss)

        # Save the best weights if the current loss is the lowest
        if test_loss < best_loss:
            best_loss = test_loss
            best_A, best_B, best_C = A.copy(), B.copy(), C.copy()

    return best_A, best_B, best_C, train_losses, test_losses

def Adam_training(data, target, A, B, C, learning_rate, epochs, beta1, beta2, epsilon):

    best_A, best_B, best_C = A.copy(), B.copy(), C.copy()
    best_loss = float('inf')
    train_losses = []
    test_losses = []

    # Initialize moment estimates for A, B, C for Adam and Nadam
    m_A, m_B, m_C = np.zeros_like(A), np.zeros_like(B), np.zeros_like(C)  # First moment
    v_A, v_B, v_C = np.zeros_like(A), np.zeros_like(B), np.zeros_like(C)  # Second moment

        # Training loop
    for epoch in range(epochs):
        # Perform SGD
        # for i in range(data.shape[1]):
            # x_sample = data[:, i:i+1]
            # t_sample = target[:, i:i+1]

        # Forward and backward propagation
        grad_A, grad_B, grad_C = backprop(data, target, A, B, C)

        # Update biased first moment estimates
        m_A = beta1 * m_A + (1 - beta1) * grad_A
        m_B = beta1 * m_B + (1 - beta1) * grad_B
        m_C = beta1 * m_C + (1 - beta1) * grad_C

        # Update biased second moment estimates
        v_A = beta2 * v_A + (1 - beta2) * (grad_A ** 2)
        v_B = beta2 * v_B + (1 - beta2) * (grad_B ** 2)
        v_C = beta2 * v_C + (1 - beta2) * (grad_C ** 2)

        # Compute bias-corrected first moment estimates
        m_A_hat = m_A / (1 - beta1 ** epoch + 1e-8)
        m_B_hat = m_B / (1 - beta1 ** epoch + 1e-8)
        m_C_hat = m_C / (1 - beta1 ** epoch + 1e-8)

        v_A_hat = v_A / (1 - beta2 ** epoch + 1e-8)
        v_B_hat = v_B / (1 - beta2 ** epoch + 1e-8)
        v_C_hat = v_C / (1 - beta2 ** epoch + 1e-8)

        # Update weights
        A -= learning_rate * m_A_hat / (np.sqrt(v_A_hat) + epsilon)
        B -= learning_rate * m_B_hat / (np.sqrt(v_B_hat) + epsilon)
        C -= learning_rate * m_C_hat / (np.sqrt(v_C_hat) + epsilon)

        # Compute training loss for the epoch
        _, _, _, train_loss = forwardprop(data, target, A, B, C)
        train_losses.append(train_loss)

        # Compute test loss for the epoch
        _, _, _, test_loss = forwardprop(data, target, A, B, C)
        test_losses.append(test_loss)

        # Save the best weights if the current loss is the lowest
        if test_loss < best_loss:
            best_loss = test_loss
            best_A, best_B, best_C = A.copy(), B.copy(), C.copy()

        return best_A, best_B, best_C, train_losses, test_losses

def Nadam_training(data, target, A, B, C, learning_rate, epochs, beta1, beta2, epsilon):

    best_A, best_B, best_C = A.copy(), B.copy(), C.copy()
    best_loss = float('inf')
    train_losses = []
    test_losses = []

    # Initialize moment estimates for A, B, C for Adam and Nadam
    m_A, m_B, m_C = np.zeros_like(A), np.zeros_like(B), np.zeros_like(C)  # First moment
    v_A, v_B, v_C = np.zeros_like(A), np.zeros_like(B), np.zeros_like(C)  # Second moment

        # Training loop
    for epoch in range(epochs):
        # Perform SGD
        # for i in range(data.shape[1]): Uncommenting this for all makes it run much slower for the same result, not sure if needed but I thought it was
        #     x_sample = data[:, i:i+1]
        #     t_sample = target[:, i:i+1]

            # Forward and backward propagation
        grad_A, grad_B, grad_C = backprop(data, target, A, B, C)

        # Update biased first moment estimates
        m_A = beta1 * m_A + (1 - beta1) * grad_A
        m_B = beta1 * m_B + (1 - beta1) * grad_B
        m_C = beta1 * m_C + (1 - beta1) * grad_C

        # Update biased second moment estimates
        v_A = beta2 * v_A + (1 - beta2) * (grad_A ** 2)
        v_B = beta2 * v_B + (1 - beta2) * (grad_B ** 2)
        v_C = beta2 * v_C + (1 - beta2) * (grad_C ** 2)

        # Compute bias-corrected first moment estimates
        m_A_hat = m_A / (1 - beta1 ** epoch + 1e-8)
        m_B_hat = m_B / (1 - beta1 ** epoch + 1e-8)
        m_C_hat = m_C / (1 - beta1 ** epoch + 1e-8)

        v_A_hat = beta1 * v_A + (1 - beta1) * grad_A / (1 - beta1 ** epoch + 1e-8)
        v_B_hat = beta1 * v_B + (1 - beta1) * grad_B / (1 - beta1 ** epoch + 1e-8)
        v_C_hat = beta1 * v_C + (1 - beta1) * grad_C / (1 - beta1 ** epoch + 1e-8)

        # Update weights
        A -= learning_rate * m_A_hat / (np.sqrt(v_A_hat) + epsilon)
        B -= learning_rate * m_B_hat / (np.sqrt(v_B_hat) + epsilon)
        C -= learning_rate * m_C_hat / (np.sqrt(v_C_hat) + epsilon)

        # Compute training loss for the epoch
        _, _, _, train_loss = forwardprop(data, target, A, B, C)
        train_losses.append(train_loss)

        # Compute test loss for the epoch
        _, _, _, test_loss = forwardprop(data, target, A, B, C)
        test_losses.append(test_loss)

        # Save the best weights if the current loss is the lowest
        if test_loss < best_loss:
            best_loss = test_loss
            best_A, best_B, best_C = A.copy(), B.copy(), C.copy()

        return best_A, best_B, best_C, train_losses, test_losses

def q3_main():

    # Load the two CSV files
    at_test_path = "/home/jackson/Documents/AERSP-597-ML/hw-2/aerothermal_test.csv"
    at_train_path = "/home/jackson/Documents/AERSP-597-ML/hw-2/aerothermal_train.csv"

    at_test_data = pd.read_csv(at_test_path)
    at_train_data = pd.read_csv(at_train_path)

    # Extract features and targets
    x_train = at_train_data.iloc[:, :-1].values.T  #(M features, N samples)
    t_train = at_train_data.iloc[:, -1].values.reshape(1, -1)  # (1, N samples)
    x_test = at_test_data.iloc[:, :-1].values.T
    t_test = at_test_data.iloc[:, -1].values.reshape(1, -1)

    # Min-max normalize the features
    x_train = min_max(x_train)
    x_test = min_max(x_test)
    t_train = min_max(t_train)
    t_test = min_max(t_test)

    # Initialize weights and biases
    np.random.seed(42)
    M = x_train.shape[0]  # Number of input features
    K = 5                # Number of hidden layer neurons
    D = 5                # Number of second hidden layer neurons
    N = t_train.shape[0]  # Number of output neurons

    A = weight_init(M, K, 'sigmoid', 'uniform') # M inputs, K outputs, input->first
    B = weight_init(K, D, 'sigmoid', 'uniform') # K inputs, D outputs, first->second
    C = weight_init(D, N, 'relu', 'uniform')    # D inputs, N outputs, second->output

    # Training parameters
    learning_rate = 0.0001
    epochs = 500
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    # Train
    A_SGD, B_SGD, C_SGD, train_loss_SGD, test_loss_SGD = SGD_training(x_train, t_train, A, B, C, learning_rate, epochs)
    A_Adam, B_Adam, C_Adam, train_loss_Adam, test_loss_Adam = Adam_training(x_train, t_train, A, B, C, learning_rate, epochs, beta1, beta2, epsilon)
    A_Nadam, B_Nadam, C_Nadam, train_loss_Nadam, test_loss_Nadam = Nadam_training(x_train, t_train, A, B, C, learning_rate, epochs, beta1, beta2, epsilon)
    print("Final Train Loss SGD:", train_loss_SGD[-1])
    print("Final Train Loss Adam:", train_loss_Adam[-1])
    print("Final Train Loss Nadam:", train_loss_Nadam[-1])
    print("Final Test Loss SGD:", test_loss_SGD[-1])
    print("Final Test Loss Adam:", test_loss_Adam[-1])
    print("Final Test Loss Nadam:", test_loss_Nadam[-1])

    # Plot training and test loss for all 3 cases
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_loss_SGD, label='Train Loss SGD', linestyle='-', color='blue')
    plt.plot(range(1, epochs + 1), test_loss_SGD, label='Test Loss SGD', linestyle='--', color='blue')
    plt.plot(range(1, epochs + 1), train_loss_Adam, label='Train Loss Adam', linestyle='-', color='green')
    plt.plot(range(1, epochs + 1), test_loss_Adam, label='Test Loss Adam', linestyle='--', color='green')
    plt.plot(range(1, epochs + 1), train_loss_Nadam, label='Train Loss Nadam', linestyle='-', color='red')
    plt.plot(range(1, epochs + 1), test_loss_Nadam, label='Test Loss Nadam', linestyle='--', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    #q1a_main()
    #gradient_check() # Question 2
    q3_main()

