import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from mpl_toolkits.mplot3d import Axes3D
import os

def load_data(train_file, test_file):
    train_data = np.load(train_file)
    test_data = np.load(test_file)
    return train_data, test_data

def prepare_data(data):
    X = []
    Y = []
    for traj in data:
        for t in range(1, traj.shape[1]):
            X.append(traj[:, t-1])  
            Y.append(traj[:, t])    
    X = np.vstack(X)  
    Y = np.vstack(Y)  
    return X, Y

def polynomial_kernel(X, Y, degree, coef0):
    return (X @ Y.T + coef0) ** degree

def train_kernel_ridge_regression(X_train, Y_train, degree, coef0, lambda_reg):
    K_train = polynomial_kernel(X_train, X_train, degree=degree, coef0=coef0)
    U, S, Vt = np.linalg.svd(K_train, full_matrices=False)
    S_inv = np.diag(1 / (S + lambda_reg))  
    alpha = Vt.T @ S_inv @ U.T @ Y_train  
    return alpha

def predict(X_test, X_train, alpha, degree, coef0):
    K_test = polynomial_kernel(X_test, X_train, degree=degree, coef0=coef0)
    Y_pred = K_test @ alpha
    return Y_pred

def compute_errors(Y_true, Y_pred):
    errors = np.linalg.norm(Y_true - Y_pred, axis=1)
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    return mean_error, max_error

def plot_trajectories(test_data, X_train, alpha, degree, coef0):
    rollout_errors = []
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(10):  
        true_trajectory = test_data[i, :, :]
        predicted_trajectory = np.zeros_like(true_trajectory)
        # Set initial condition for the predicted trajectory
        predicted_trajectory[:, 0] = true_trajectory[:, 0]  

        for t in range(1, true_trajectory.shape[1]):
            K_rollout = polynomial_kernel(predicted_trajectory[:, t-1].reshape(1, -1), X_train, degree=degree, coef0=coef0)
            predicted_trajectory[:, t] = (K_rollout @ alpha).flatten()

        # Compute Roll-Out Errors for this trajectory
        rollout_errors.append(np.linalg.norm(true_trajectory - predicted_trajectory, axis=0))

        # Plot true vs predicted trajectory
        ax.plot(true_trajectory[0], true_trajectory[1], true_trajectory[2], color='C'+str(i), label=f'True Traj {i+1}' if i == 0 else "")
        ax.plot(predicted_trajectory[0], predicted_trajectory[1], predicted_trajectory[2], color='C'+str(i), linestyle='--', label=f'Pred Traj {i+1}' if i == 0 else "")

    # Compute mean and max roll-out error across all trajectories
    rollout_errors = np.array(rollout_errors)
    mean_rollout_error = np.mean(rollout_errors)
    max_rollout_error = np.max(rollout_errors)
    print(f"Mean Roll-Out Error (e_r): {mean_rollout_error:.4f}")
    print(f"Max Roll-Out Error (e_r): {max_rollout_error:.4f}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('True vs Predicted 3D Trajectories (Kernel Polynomial Model)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.show()

def main():
    # Define file paths
    work_folder_path = 'C:/Users/jfeze/Documents/Documents/Graduate School/2. Spring 2025/AERSP-597-ML/hw-1/'
    train_file = work_folder_path + 'train_dyn_hw1.npy'
    test_file = work_folder_path + 'test_dyn_hw1.npy'

    # Load data
    train_data, test_data = load_data(train_file, test_file)

    # Prepare training and test data
    X_train, Y_train = prepare_data(train_data)
    X_test, Y_test = prepare_data(test_data)

    # Train Kernel Ridge Regression Model with Polynomial Kernel
    degree = 6
    coef0 = 3
    lambda_reg = 1e-2
    alpha = train_kernel_ridge_regression(X_train, Y_train, degree, coef0, lambda_reg)

    # Predict on test data
    Y_pred = predict(X_test, X_train, alpha, degree, coef0)

    # Compute One-Step Error
    mean_one_step_error, max_one_step_error = compute_errors(Y_test, Y_pred)
    print(f"Mean One-Step Error (e_o): {mean_one_step_error:.4f}")
    print(f"Max One-Step Error (e_o): {max_one_step_error:.4f}")

    # Plot true vs predicted trajectories and compute roll-out errors
    plot_trajectories(test_data, X_train, alpha, degree, coef0)

if __name__ == "__main__":
    main()