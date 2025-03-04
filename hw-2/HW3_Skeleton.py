import numpy as np

epsilon = 1e-4
M = 100
K = 50
D = 30
N = 10
nCheck = 1000

def swish(x):
    return x / (1+np.exp(-x))


def forwardprop(x, t, A, B, C):

    # ---------- make your implementation here -------------
    y = 0
    z = 0
    h = 0  # Prediction
    E = 0  # Error
    # -------------------------------------------------

    return y, z, h, E


def backprop(x, t, A, B, C):

    y, z, h, J = forwardprop(x, t, A, B, C)

    # ---------- make your implementation here -------------
    grad_C = 0
    grad_B = 0
    grad_A = 0
    # -------------------------------------------------

    return grad_A, grad_B, grad_C

def gradient_check():

    A = np.random.rand(K, M+1)*0.1-0.05
    B = np.random.rand(D, K+1)*0.1-0.05
    C = np.random.rand(N, D+1)*0.1-0.05
    x = np.random.rand(M, 1)*0.1-0.05
    t = np.random.rand(N, 1)*0.2-0.1

    grad_A, grad_B, grad_C = backprop(x, A, B, C)
    errA, errB, errC = [], [], []

    for i in range(1000):

        # ---------- make your implementation here -------------
        idx_x, idx_y = 0, 0

        # numerical gradients at (idx_x, idx_y)
        numerical_grad_A = 0
        errA.append(np.abs(grad_A[idx_x, idx_y] - numerical_grad_A))

        numerical_grad_B = 0
        errB.append(np.abs(grad_B[idx_x, idx_y] - numerical_grad_B))

        numerical_grad_C = 0
        errC.append(np.abs(grad_C[idx_x, idx_y] - numerical_grad_C))
        # -------------------------------------------------

    print('Gradient checking A, MAE: {0:0.8f}'.format(np.mean(errA)))
    print('Gradient checking B, MAE: {0:0.8f}'.format(np.mean(errB)))
    print('Gradient checking C, MAE: {0:0.8f}'.format(np.mean(errC)))

if __name__ == '__main__':
    gradient_check()