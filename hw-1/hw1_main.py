import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read data from a file
def read_data(file_path):
    data = pd.read_csv(file_path)
    print(data.head())       # Show the first few rows
    print(data.info())       # Overview of dataset structure
    print(data.describe())   # Summary statistics
    return data

# Step 2: Manipulate the data
def manipulate_data(data):
    ones = np.ones((data.shape[0], 1))
    biased_data = np.hstack((ones, data))
    return biased_data

# Step 3: Normalize the data
def normalize_data(data):
    normalized_data = data.copy()
    for col in data.columns:
        min_val = data[col].min()
        max_val = data[col].max()
        normalized_data[col] = (data[col] - min_val) / (max_val - min_val)
    return normalized_data

# Step 4: Plot the data
def plot_data(data):
    # Example plot: scatter plot of columns 'A' vs 'B'
    plt.scatter(data['A'], data['B'])
    plt.xlabel('A')
    plt.ylabel('B')
    plt.title('Scatter plot of A vs B')
    plt.show()

# Main function to execute the steps
def main():
    fd_test_file = 'flight_data_test.csv'
    fd_train_file = 'flight_data_train.csv'
    fd_test_data = read_data(fd_test_file)
    fd_train_data = read_data(fd_train_file)
    
    # Uncomment the following lines to manipulate and plot the data
    #manipulated_test_data = manipulate_data(fd_test_data)
    #plot_data(manipulated_test_data)

if __name__ == "__main__":
    main()