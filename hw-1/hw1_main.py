import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Read data from a file
def read_data(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    data = pd.read_csv(file_path)
    # print(data.head())       # Show the first few rows
    # print(data.info())       # Overview of dataset structure
    # print(data.describe())   # Summary statistics
    return data

# Normalize the data
def normalize_data(data, ouput):
    normalized_data = data.copy()
    for col in data.columns:
        if col != ouput:
            mean_val = data[col].mean()
            std_val = data[col].std()
            # Avoid blowups
            normalized_data[col] = (data[col] - mean_val) / (std_val + 1e-6)
    return normalized_data

# Generate polynomial features
def generate_polynomial_feats(data, degree):
    poly_feats = np.ones((data.shape[0], 1))  # Start with bias term
    for d in range(1, degree + 1):
        for col in data.columns:
            poly_feats = np.hstack((poly_feats, np.power(data[col].values.reshape(-1, 1), d)))
    return poly_feats

# Pseudo-inverse linear regression using SVD
def pseudo_inverse_linear_regression(X, y):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S_inv = np.diag(1 / (S+1e-4))  # Avoid blowups
    X_pseudo_inv = Vt.T @ S_inv @ U.T
    weights = X_pseudo_inv @ y
    return weights

# Pseudo-inverse ridge regression using SVD
def pseudo_inverse_ridge_regression(X, y, lambda_val):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S_inv = np.diag(S / (S**2 + lambda_val))
    X_pseudo_inv = Vt.T @ S_inv @ U.T
    weights = X_pseudo_inv @ y
    return weights

# Calculate RMSE
def calculate_rmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse

# 10-fold Cross-Validation
def cross_validation(X, y, degree, ln_lambda_vals):
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    fold_size = len(X) // 10
    avg_rmse = []

    for ln_lambda in ln_lambda_vals:
        lambda_val = np.exp(ln_lambda)
        fold_rmse = []

        for i in range(10):
            val_indices = indices[i*fold_size:(i+1)*fold_size]
            train_indices = np.concatenate([indices[:i*fold_size], indices[(i+1)*fold_size:]])

            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]

            X_train_poly = generate_polynomial_feats(pd.DataFrame(X_train), degree)
            X_val_poly = generate_polynomial_feats(pd.DataFrame(X_val), degree)

            weights = pseudo_inverse_ridge_regression(X_train_poly, y_train, lambda_val)
            y_val_pred = np.dot(X_val_poly, weights)
            val_rmse = calculate_rmse(y_val, y_val_pred)
            fold_rmse.append(val_rmse)

        avg_rmse.append(np.mean(fold_rmse))
    return avg_rmse

# Akaike Information Criteria (AIC)
def calculate_aic(X, y, degree, ln_lambda_vals):
    N = X.shape[0]
    aic_vals = []

    for ln_lambda in ln_lambda_vals:
        lambda_val = np.exp(ln_lambda)
        X_poly = generate_polynomial_feats(pd.DataFrame(X), degree)
        weights = pseudo_inverse_ridge_regression(X_poly, y, lambda_val)
        y_pred = np.dot(X_poly, weights)
        ED_w = np.sum((y - y_pred) ** 2)
        gamma = np.trace(np.linalg.inv(X_poly.T @ X_poly + lambda_val * np.eye(X_poly.shape[1])) @ (X_poly.T @ X_poly))
        aic = N * np.log(ED_w / N) + gamma
        aic_vals.append(aic)
    return aic_vals

# Bayesian Information Criteria (BIC)
def calculate_bic(X, y, degree, ln_lambda_vals):
    N = X.shape[0]
    bic_vals = []

    for ln_lambda in ln_lambda_vals:
        lambda_val = np.exp(ln_lambda)
        X_poly = generate_polynomial_feats(pd.DataFrame(X), degree)
        weights = pseudo_inverse_ridge_regression(X_poly, y, lambda_val)
        y_pred = np.dot(X_poly, weights)
        ED_w = np.sum((y - y_pred) ** 2)
        gamma = np.trace(np.linalg.inv(X_poly.T @ X_poly + lambda_val * np.eye(X_poly.shape[1])) @ (X_poly.T @ X_poly))
        bic = N * np.log(ED_w / N) + gamma * np.log(N)
        bic_vals.append(bic)

    return bic_vals

# Main function for Question 1.1
def q1_1():
    work_folder_path = 'C:/Users/jfeze/Documents/Documents/Graduate School/2. Spring 2025/AERSP-597-ML/hw-1/'
    fd_test_file  = work_folder_path + 'flight_data_test.csv'
    fd_train_file = work_folder_path + 'flight_data_train.csv'
    
    fd_test_data = read_data(fd_test_file)
    fd_train_data = read_data(fd_train_file)

    # Singling out the output column
    ouput = fd_train_data.columns[-1]
    X_train = fd_train_data.iloc[:, :-1]
    y_train = fd_train_data.iloc[:, -1]
    X_test = fd_test_data.iloc[:, :-1]
    y_test = fd_test_data.iloc[:, -1]

     # Normalize the data
    X_train = normalize_data(X_train, ouput)
    X_test = normalize_data(X_test, ouput)

    max_degree = 6
    train_errors = []
    test_errors = []

    for degree in range(1, max_degree + 1):
        X_train_poly = generate_polynomial_feats(X_train, degree)
        X_test_poly = generate_polynomial_feats(X_test, degree)

        print(f"Degree: {degree}")
        print(f"X_train_poly shape: {X_train_poly.shape}")
        print(f"X_test_poly shape: {X_test_poly.shape}")

        weights = pseudo_inverse_linear_regression(X_train_poly, y_train)

        y_train_pred = np.dot(X_train_poly, weights)
        y_test_pred = np.dot(X_test_poly, weights)

        train_rmse = calculate_rmse(y_train, y_train_pred)
        test_rmse = calculate_rmse(y_test, y_test_pred)

        train_errors.append(train_rmse)
        test_errors.append(test_rmse)

    plt.plot(range(1, max_degree + 1), train_errors, marker='o',label='Training RMSE')
    plt.plot(range(1, max_degree + 1), test_errors, marker='o',label='Test RMSE')
    plt.xlabel('Order of Polynomial feats (m)')
    plt.ylabel('Root Mean Square Error (RMSE)')
    plt.title('Training and Test RMSE vs Order of Polynomial feats')
    plt.legend()
    plt.show()

# Main function for Question 1.2
def q1_2():
    work_folder_path = 'C:/Users/jfeze/Documents/Documents/Graduate School/2. Spring 2025/AERSP-597-ML/hw-1/'
    fd_test_file  = work_folder_path + 'flight_data_test.csv'
    fd_train_file = work_folder_path + 'flight_data_train.csv'
    
    fd_test_data = read_data(fd_test_file)
    fd_train_data = read_data(fd_train_file)

    # Singling out the output column
    ouput = fd_train_data.columns[-1]
    X_train = fd_train_data.iloc[:, :-1]
    y_train = fd_train_data.iloc[:, -1]
    X_test = fd_test_data.iloc[:, :-1]
    y_test = fd_test_data.iloc[:, -1]

     # Normalize the data
    X_train = normalize_data(X_train, ouput)
    X_test = normalize_data(X_test, ouput)

    degree = 6
    train_errors = []
    test_errors = []
    ln_lambda_vals = np.arange(-30, 11, 1)

    X_train_poly = generate_polynomial_feats(X_train, degree)
    X_test_poly = generate_polynomial_feats(X_test, degree)

    for ln_lambda in ln_lambda_vals:

        lambda_val = np.exp(ln_lambda)
        weights = pseudo_inverse_ridge_regression(X_train_poly, y_train, lambda_val)

        y_train_pred = np.dot(X_train_poly, weights)
        y_test_pred = np.dot(X_test_poly, weights)

        train_rmse = calculate_rmse(y_train, y_train_pred)
        test_rmse = calculate_rmse(y_test, y_test_pred)

        train_errors.append(train_rmse)
        test_errors.append(test_rmse)

    plt.plot(ln_lambda_vals, train_errors,marker='o', label='Training RMSE')
    plt.plot(ln_lambda_vals, test_errors, marker='o', label='Test RMSE')
    plt.xlabel('Natural Log of Lambda Value (ln(λ))')
    plt.ylabel('Root Mean Square Error (RMSE)')
    plt.title('Training and Test RMSE vs Natural Log of Lambda')
    plt.legend()
    plt.show()


# Main function for Question 1.3
def q1_3():
    work_folder_path = 'C:/Users/jfeze/Documents/Documents/Graduate School/2. Spring 2025/AERSP-597-ML/hw-1/'
    fd_train_file = work_folder_path + 'flight_data_train.csv'
    
    fd_train_data = read_data(fd_train_file)

    # Singling out the output column
    output_col = fd_train_data.columns[-1]
    X_train = fd_train_data.iloc[:, :-1].values
    y_train = fd_train_data.iloc[:, -1].values

    # Normalize the data
    X_train = normalize_data(pd.DataFrame(X_train), output_col).values

    degree = 6
    ln_lambda_vals = np.arange(-30, 11, 1)

    # 10-fold Cross-Validation
    cv_10fold_rmse = cross_validation(X_train, y_train, degree, ln_lambda_vals)

    # Akaike and Bayesian Information Criteria
    aic_vals = calculate_aic(X_train, y_train, degree, ln_lambda_vals)
    bic_vals = calculate_bic(X_train, y_train, degree, ln_lambda_vals)

    # Find the best lambda values
    min_cv_index = np.argmin(cv_10fold_rmse)
    min_aic_index = np.argmin(aic_vals)
    min_bic_index = np.argmin(bic_vals)

    best_lambda_cv = np.exp(ln_lambda_vals[min_cv_index])
    best_lambda_aic = np.exp(ln_lambda_vals[min_aic_index])
    best_lambda_bic = np.exp(ln_lambda_vals[min_bic_index])

    print(f"Best lambda (CV): {np.log(best_lambda_cv)}")
    print(f"Best lambda (AIC): {np.log(best_lambda_aic)}")
    print(f"Best lambda (BIC): {np.log(best_lambda_bic)}")

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(ln_lambda_vals, cv_10fold_rmse, marker='o', label='10-fold CV RMSE')
    plt.xlabel('Natural Log of Lambda Value (ln(λ))')
    plt.ylabel('10-fold CV RMSE')
    plt.title('10-fold CV RMSE vs Natural Log of Lambda')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(ln_lambda_vals, aic_vals, marker='o', label='AIC')
    plt.plot(ln_lambda_vals, bic_vals, marker='o', label='BIC')
    plt.xlabel('Natural Log of Lambda Value (ln(λ))')
    plt.ylabel('Metric Value')
    plt.title('AIC and BIC vs Natural Log of Lambda')
    plt.legend()
    plt.show()

def main():
    #q1_1()
    #q1_2()
    q1_3()

if __name__ == "__main__":
    main()