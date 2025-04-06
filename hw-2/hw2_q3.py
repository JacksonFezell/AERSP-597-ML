import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ACTIVATION FUNCTIONS AND DERIVATIVES

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def swish(x):
    output = x * sigmoid(x)
    #print(f"Swish output: {output}")
    return output

def relu(x):
    #print(f"ReLU maximum input: {np.maximum(0, x)}")
    return np.maximum(0, x)

def swish_deriv(x):
    sig = sigmoid(x)
    return sig + x * sig * (1 - sig)

def relu_deriv(y):
    return np.where(y > 0, 1, 0)



# PROPAGATION FUNCTIONS

def forwardprop(x, t, A, B, C):
    x_bias = np.vstack([x, np.ones((1, x.shape[1]))]) 
    pre_y = A @ x_bias
    y = swish(pre_y)

    y_bias = np.vstack([y, np.ones((1, y.shape[1]))])
    pre_z = B @ y_bias
    z = swish(pre_z)

    z_bias = np.vstack([z, np.ones((1, z.shape[1]))])
    pre_h = C @ z_bias
    h = relu(pre_h)

    E = 0.5 * np.mean((h - t) ** 2)  # Mean Squared Error
    #print(f"MSE: {E}")
    E_grad = 2 * (h - t) / h.shape[1]
    #print(f"Gradient of MSE: {E_grad}")
    return pre_y, y, pre_z, z, pre_h, h, E

def backprop(x, t, A, B, C):
    pre_y, y, pre_z, z, pre_h, h, E = forwardprop(x, t, A, B, C)
    
    # Output layer weights
    del_C = (h - t) * (relu_deriv(pre_h))
    grad_C = del_C @ np.vstack([z, np.ones((1, z.shape[1]))]).T

    # Second hidden layer weights
    del_B = (C[:, :-1].T @ del_C) * swish_deriv(pre_z) 
    grad_B = del_B @ np.vstack([y, np.ones((1, y.shape[1]))]).T 

    # First hidden layer weights
    del_A = (B[:, :-1].T @ del_B) * swish_deriv(pre_y) 
    grad_A = del_A @ np.vstack([x, np.ones((1, x.shape[1]))]).T 
    
    # print("Mean grad_A:", np.mean(np.abs(grad_A)))
    # print("Mean grad_B:", np.mean(np.abs(grad_B)))
    # print("Mean grad_C:", np.mean(np.abs(grad_C)))

    
    return grad_A, grad_B, grad_C


# TRAINING PROCESSING FUNCTIONS

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
    if activation in ('relu', 'swish'): 
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


# TRAINING FUNCTIONS

def SGD_training(data, target, data_test, target_test, A, B, C, learning_rate, epochs):

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
        # print("Gradients A:", grad_A)
        # print("Gradients B:", grad_B)
        # print("Gradients C:", grad_C)


        # SGD fixed learning rate
        A -= learning_rate * grad_A
        B -= learning_rate * grad_B
        C -= learning_rate * grad_C

        # Compute training loss for the epoch
        _, _, _, _, _, _, train_loss = forwardprop(data, target, A, B, C)
        train_losses.append(train_loss)

        # Compute test loss for the epoch
        _, _, _, _, _, _, test_loss = forwardprop(data_test, target_test, A, B, C)
        test_losses.append(test_loss)

        # Save the best weights if the current loss is the lowest
        if test_loss < best_loss:
            best_loss = test_loss
            best_A, best_B, best_C = A.copy(), B.copy(), C.copy()

    return best_A, best_B, best_C, train_losses, test_losses

def Adam_training(data, target, data_test, target_test, A, B, C, learning_rate, epochs, beta1, beta2, epsilon):

    best_A, best_B, best_C = A.copy(), B.copy(), C.copy()
    best_loss = float('inf')
    train_losses = []
    test_losses = []

    # Initialize moment estimates for A, B, C for Adam and Nadam
    v_A, v_B, v_C = np.zeros_like(A), np.zeros_like(B), np.zeros_like(C)  # First moment
    g_A, g_B, g_C = np.zeros_like(A), np.zeros_like(B), np.zeros_like(C)  # Second moment

        # Training loop
    for epoch in range(epochs):
        # Perform SGD
        # for i in range(data.shape[1]):
            # x_sample = data[:, i:i+1]
            # t_sample = target[:, i:i+1]

        # Forward and backward propagation
        grad_A, grad_B, grad_C = backprop(data, target, A, B, C)

        # Update biased first moment estimates
        v_A = beta1 * v_A + (1 - beta1) * grad_A
        v_B = beta1 * v_B + (1 - beta1) * grad_B
        v_C = beta1 * v_C + (1 - beta1) * grad_C

        # Update biased second moment estimates
        g_A = beta2 * g_A + (1 - beta2) * (grad_A ** 2)
        g_B = beta2 * g_B + (1 - beta2) * (grad_B ** 2)
        g_C = beta2 * g_C + (1 - beta2) * (grad_C ** 2)

        # Compute bias-corrected first moment estimates
        v_A_hat = v_A / (1 - beta1 ** (epoch + 1))
        v_B_hat = v_B / (1 - beta1 ** (epoch + 1))
        v_C_hat = v_C / (1 - beta1 ** (epoch + 1))

        g_A_hat = g_A / (1 - beta2 ** (epoch + 1))
        g_B_hat = g_B / (1 - beta2 ** (epoch + 1))
        g_C_hat = g_C / (1 - beta2 ** (epoch + 1))

        # Update weights
        A -= learning_rate * v_A_hat / (np.sqrt(g_A_hat) + epsilon)
        B -= learning_rate * v_B_hat / (np.sqrt(g_B_hat) + epsilon)
        C -= learning_rate * v_C_hat / (np.sqrt(g_C_hat) + epsilon)

        # Compute training loss for the epoch
        _, _, _,  _, _, _,train_loss = forwardprop(data, target, A, B, C)
        train_losses.append(train_loss)

        # Compute test loss for the epoch
        _, _, _,  _, _, _,test_loss = forwardprop(data_test, target_test, A, B, C)
        test_losses.append(test_loss)

        # Save the best weights if the current loss is the lowest
        if test_loss < best_loss:
            best_loss = test_loss
            best_A, best_B, best_C = A.copy(), B.copy(), C.copy()

    return best_A, best_B, best_C, train_losses, test_losses

def Nadam_training(data, target, data_test, target_test, A, B, C, learning_rate, epochs, beta1, beta2, epsilon):

    best_A, best_B, best_C = A.copy(), B.copy(), C.copy()
    best_loss = float('inf')
    train_losses = []
    test_losses = []

    # Initialize moment estimates for A, B, C for Adam and Nadam
    v_A, v_B, v_C = np.zeros_like(A), np.zeros_like(B), np.zeros_like(C)  # First moment
    g_A, g_B, g_C = np.zeros_like(A), np.zeros_like(B), np.zeros_like(C)  # Second moment

        # Training loop
    for epoch in range(epochs):
        # Perform SGD
        # for i in range(data.shape[1]): Uncommenting this for all makes it run much slower for the same result, not sure if needed but I thought it was
        #     x_sample = data[:, i:i+1]
        #     t_sample = target[:, i:i+1]

            # Forward and backward propagation
        grad_A, grad_B, grad_C = backprop(data, target, A, B, C)

        # Update biased first moment estimates
        v_A = beta1 * v_A + (1 - beta1) * grad_A
        v_B = beta1 * v_B + (1 - beta1) * grad_B
        v_C = beta1 * v_C + (1 - beta1) * grad_C

        # Update biased second moment estimates
        g_A = beta2 * g_A + (1 - beta2) * (grad_A ** 2)
        g_B = beta2 * g_B + (1 - beta2) * (grad_B ** 2)
        g_C = beta2 * g_C + (1 - beta2) * (grad_C ** 2)

        # Compute bias-corrected first moment estimates
        v_A_hat = v_A / (1 - beta1 ** (epoch + 1))
        v_B_hat = v_B / (1 - beta1 ** (epoch + 1))
        v_C_hat = v_C / (1 - beta1 ** (epoch + 1))
    
        g_A_hat = g_A / (1 - beta2 ** (epoch + 1))
        g_B_hat = g_B / (1 - beta2 ** (epoch + 1))
        g_C_hat = g_C / (1 - beta2 ** (epoch + 1))

        v_A_nesterov = beta1 * v_A_hat + (1 - beta1) * grad_A / (1 - beta1 ** (epoch + 1))
        v_B_nesterov = beta1 * v_B_hat + (1 - beta1) * grad_B / (1 - beta1 ** (epoch + 1))
        v_C_nesterov = beta1 * v_C_hat + (1 - beta1) * grad_C / (1 - beta1 ** (epoch + 1))
        
        # Update weights using Nesterov momentum
        A -= learning_rate * v_A_nesterov / (np.sqrt(g_A_hat) + epsilon)
        B -= learning_rate * v_B_nesterov / (np.sqrt(g_B_hat) + epsilon)
        C -= learning_rate * v_C_nesterov / (np.sqrt(g_C_hat) + epsilon)


        # Compute training loss for the epoch
        _, _, _,  _, _, _,train_loss = forwardprop(data, target, A, B, C)
        train_losses.append(train_loss)

        # Compute test loss for the epoch
        _, _, _,  _, _, _,test_loss = forwardprop(data_test, target_test, A, B, C)
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
    #Snp.random.seed(13)
    M = x_train.shape[0]  # Number of input features
    K = 5                # Number of hidden layer neurons
    D = 5                # Number of second hidden layer neurons
    N = t_train.shape[0]  # Number of output neurons

    A = weight_init(M, K, 'swish', 'uniform') # M inputs, K outputs, input->first
    B = weight_init(K, D, 'swish', 'uniform') # K inputs, D outputs, first->second
    C = weight_init(D, N, 'relu', 'uniform')    # D inputs, N outputs, second->output

    # Training parameters
    learning_rate = 0.0001
    epochs = 1000
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    # Train
    A_SGD, B_SGD, C_SGD, train_loss_SGD, test_loss_SGD = SGD_training(x_train, t_train, x_test, t_test, A, B, C, learning_rate, epochs)
    A_Adam, B_Adam, C_Adam, train_loss_Adam, test_loss_Adam = Adam_training(x_train, t_train, x_test, t_test, A, B, C, learning_rate, epochs, beta1, beta2, epsilon)
    A_Nadam, B_Nadam, C_Nadam, train_loss_Nadam, test_loss_Nadam = Nadam_training(x_train, t_train, x_test, t_test, A, B, C, learning_rate, epochs, beta1, beta2, epsilon)
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

    q3_main()
    
    
    
'''
Old prop functions

def forwardprop(x, t, A, B, C):
    x_bias = np.vstack([x, np.ones((1, x.shape[1]))])
    y = swish(A @ x_bias)
    y_bias = np.vstack([y, np.ones((1, y.shape[1]))])
    z = swish(B @ y_bias)
    z_bias = np.vstack([z, np.ones((1, z.shape[1]))])
    h = relu(C @ z_bias)
    E = 0.5 * np.sum((h - t) ** 2)  # Mean Squared Error
    return y, z, h, E

def backprop(x, t, A, B, C):
    y, z, h, E = forwardprop(x, t, A, B, C)
    
    delta_C = (h - t) * (h > 0)  # Derivative of ReLU
    grad_C = delta_C @ np.vstack([z, np.ones((1, z.shape[1]))]).T
    
    delta_B = (C[:, :-1].T @ delta_C) * (z * (1 - z))
    grad_B = delta_B @ np.vstack([y, np.ones((1, y.shape[1]))]).T
    
    delta_A = (B[:, :-1].T @ delta_B) * (y * (1 - y))
    grad_A = delta_A @ np.vstack([x, np.ones((1, x.shape[1]))]).T
    
    return grad_A, grad_B, grad_C
'''