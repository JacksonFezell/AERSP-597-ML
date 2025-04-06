import numpy as np
import matplotlib.pyplot as plt

# Load data
train_data = np.load("/home/jackson/Documents/AERSP-597-ML/hw-2/train_dyn_hw2.npy")  # shape (40, 3, 201)
test_data  = np.load("/home/jackson/Documents/AERSP-597-ML/hw-2/test_dyn_hw2.npy")# shape (60, 3, 201)

# Time step
dt = 0.04

# Pairs
train_inputs = []
train_targets = []
num_train_traj = train_data.shape[0]
n_timesteps = train_data.shape[2]

for traj in range(num_train_traj):
    for t in range(1, n_timesteps - 1):
        pos_current = train_data[traj, :, t].reshape(3, 1)
        pos_prev    = train_data[traj, :, t-1].reshape(3, 1)
        input_vec = np.vstack((pos_current, pos_prev))
        train_inputs.append(input_vec)
        train_targets.append(train_data[traj, :, t+1].reshape(3, 1))

train_inputs = np.hstack(train_inputs)
train_targets = np.hstack(train_targets)

test_inputs = []
test_targets = []
num_test_traj = test_data.shape[0]
for traj in range(num_test_traj):
    for t in range(1, n_timesteps - 1):
        pos_current = test_data[traj, :, t].reshape(3, 1)
        pos_prev    = test_data[traj, :, t-1].reshape(3, 1)
        input_vec = np.vstack((pos_current, pos_prev))
        test_inputs.append(input_vec)
        test_targets.append(test_data[traj, :, t+1].reshape(3, 1))

test_inputs = np.hstack(test_inputs)
test_targets = np.hstack(test_targets)

# Layers and Neurons
input_dim   = 6
hidden_dim1 = 64
hidden_dim2 = 64
hidden_dim3 = 32
output_dim  = 3

def xavier_init(fan_in, fan_out):
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_out, fan_in))

def he_init(fan_in, fan_out):
    limit = np.sqrt(6 / fan_in)
    return np.random.uniform(-limit, limit, (fan_out, fan_in))    


W1 = xavier_init(input_dim, hidden_dim1)
b1 = np.zeros((hidden_dim1, 1))
W2 = xavier_init(hidden_dim1, hidden_dim2)
b2 = np.zeros((hidden_dim2, 1))
W3 = he_init(hidden_dim1, hidden_dim2)
b3 = np.zeros((hidden_dim2, 1))
W4 = xavier_init(hidden_dim3, output_dim)
b4 = np.zeros((output_dim, 1))

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def relu_deriv(y):
    return np.where(y > 0, 1, 0)

def forward(x):
    current_position = x[:3, :]
    z1 = W1 @ x + b1
    a1 = tanh(z1)
    z2 = W2 @ a1 + b2
    a2 = tanh(z2)
    z3 = W3 @ a2 + b3
    a3 = relu(z3)
    z4 = W3 @ a3 + b4
    y_pred = current_position + z4
    cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3, "a3": a3, "z4": z4}
    return y_pred, cache

def compute_loss(y_pred, y_true):
    m = y_true.shape[1]
    loss = np.sum((y_pred - y_true)**2) / (2 * m)
    return loss

def backward(y_pred, y_true, cache):
    m = y_true.shape[1]
    dy = (y_pred - y_true) / m


    dz4 = dy
    dW4 = dz4 @ cache["a3"].T
    db4 = np.sum(dz4, axis=1, keepdims=True)
    
    da3 = W4.T @ dz4
    dz3 = da3 * relu_deriv(cache["z3"])
    dW3 = dz3 @ cache["a2"].T
    db3 = np.sum(dz3, axis=1, keepdims=True)

    da2 = W3.T @ dz3
    dz2 = da2 * tanh_deriv(cache["z2"])
    dW2 = dz2 @ cache["a1"].T
    db2 = np.sum(dz2, axis=1, keepdims=True)
    
    da1 = W2.T @ dz2
    dz1 = da1 * tanh_deriv(cache["z1"])
    dW1 = dz1 @ cache["x"].T
    db1 = np.sum(dz1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1, "db1": db1,
             "dW2": dW2, "db2": db2,
             "dW3": dW3, "db3": db3,
             "dW4": dW4, "db4": db4}
    return grads

learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

mW1, vW1 = np.zeros_like(W1), np.zeros_like(W1)
mb1, vb1 = np.zeros_like(b1), np.zeros_like(b1)
mW2, vW2 = np.zeros_like(W2), np.zeros_like(W2)
mb2, vb2 = np.zeros_like(b2), np.zeros_like(b2)
mW3, vW3 = np.zeros_like(W3), np.zeros_like(W3)
mb3, vb3 = np.zeros_like(b3), np.zeros_like(b3)
mW4, vW4 = np.zeros_like(W4), np.zeros_like(W4)
mb4, vb4 = np.zeros_like(b4), np.zeros_like(b4)

n_epochs = 2000
loss_history_train = []
loss_history_test  = []
timestep = 0

# Training loop
for epoch in range(1, n_epochs+1):
    y_pred_train, cache = forward(train_inputs)
    loss_train = compute_loss(y_pred_train, train_targets)
    loss_history_train.append(loss_train)
    
    y_pred_test, _ = forward(test_inputs)
    loss_test = compute_loss(y_pred_test, test_targets)
    loss_history_test.append(loss_test)
    
    grads = backward(y_pred_train, train_targets, cache)
    
    timestep += 1

    mW1 = beta1 * mW1 + (1 - beta1) * grads["dW1"]
    vW1 = beta2 * vW1 + (1 - beta2) * (grads["dW1"]**2)
    mW1_hat = mW1 / (1 - beta1**timestep)
    vW1_hat = vW1 / (1 - beta2**timestep)
    W1 = W1 - learning_rate * mW1_hat / (np.sqrt(vW1_hat) + epsilon)
    
    mb1 = beta1 * mb1 + (1 - beta1) * grads["db1"]
    vb1 = beta2 * vb1 + (1 - beta2) * (grads["db1"]**2)
    mb1_hat = mb1 / (1 - beta1**timestep)
    vb1_hat = vb1 / (1 - beta2**timestep)
    b1 = b1 - learning_rate * mb1_hat / (np.sqrt(vb1_hat) + epsilon)
    
    mW2 = beta1 * mW2 + (1 - beta1) * grads["dW2"]
    vW2 = beta2 * vW2 + (1 - beta2) * (grads["dW2"]**2)
    mW2_hat = mW2 / (1 - beta1**timestep)
    vW2_hat = vW2 / (1 - beta2**timestep)
    W2 = W2 - learning_rate * mW2_hat / (np.sqrt(vW2_hat) + epsilon)
    
    mb2 = beta1 * mb2 + (1 - beta1) * grads["db2"]
    vb2 = beta2 * vb2 + (1 - beta2) * (grads["db2"]**2)
    mb2_hat = mb2 / (1 - beta1**timestep)
    vb2_hat = vb2 / (1 - beta2**timestep)
    b2 = b2 - learning_rate * mb2_hat / (np.sqrt(vb2_hat) + epsilon)
    
    mW3 = beta1 * mW3 + (1 - beta1) * grads["dW3"]
    vW3 = beta2 * vW3 + (1 - beta2) * (grads["dW3"]**2)
    mW3_hat = mW3 / (1 - beta1**timestep)
    vW3_hat = vW3 / (1 - beta2**timestep)
    W3 = W3 - learning_rate * mW3_hat / (np.sqrt(vW3_hat) + epsilon)
    
    mb3 = beta1 * mb3 + (1 - beta1) * grads["db3"]
    vb3 = beta2 * vb3 + (1 - beta2) * (grads["db3"]**2)
    mb3_hat = mb3 / (1 - beta1**timestep)
    vb3_hat = vb3 / (1 - beta2**timestep)
    b3 = b3 - learning_rate * mb3_hat / (np.sqrt(vb3_hat) + epsilon)

    mW4 = beta1 * mW4 + (1 - beta1) * grads["dW4"]
    vW4 = beta2 * vW4 + (1 - beta2) * (grads["dW4"]**2)
    mW4_hat = mW4 / (1 - beta1**timestep)
    vW4_hat = vW4 / (1 - beta2**timestep)
    W4 = W4 - learning_rate * mW4_hat / (np.sqrt(vW4_hat) + epsilon)
    
    mb4 = beta1 * mb4 + (1 - beta1) * grads["db4"]
    vb4 = beta2 * vb4 + (1 - beta2) * (grads["db4"]**2)
    mb4_hat = mb4 / (1 - beta1**timestep)
    vb4_hat = vb4 / (1 - beta2**timestep)
    b4 = b4 - learning_rate * mb4_hat / (np.sqrt(vb4_hat) + epsilon)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{n_epochs}, Training Loss: {loss_train:.6f}, Test Loss: {loss_test:.6f}")

# One-Step Error
one_step_errors = []
one_step_preds = []
for traj in range(num_test_traj):
    pred_traj = np.zeros((3, n_timesteps))
    pred_traj[:, 0:2] = test_data[traj, :, 0:2]
    errors = []
    for t in range(1, n_timesteps - 1):
        pos_current = test_data[traj, :, t].reshape(3, 1)
        pos_prev    = test_data[traj, :, t-1].reshape(3, 1)
        x_in = np.vstack((pos_current, pos_prev))
        true_next = test_data[traj, :, t+1].reshape(3, 1)
        y_pred, _ = forward(x_in)
        pred_traj[:, t+1] = y_pred.flatten()
        errors.append(np.sum((y_pred - true_next)**2))
    one_step_errors.append(np.mean(errors))
    one_step_preds.append(pred_traj)

one_step_error_avg = np.mean(one_step_errors)
one_step_error_max = np.max(one_step_errors)
print(f"\nOne-Step Error (MSE) mean: {one_step_error_avg:.6f}")
print(f"One-Step Error (MSE) max: {one_step_error_max:.6f}")

# Roll-Out Error
rollout_errors = []
rollout_trajs = []
for traj in range(num_test_traj):
    pos_prev    = test_data[traj, :, 0].reshape(3, 1)
    pos_current = test_data[traj, :, 1].reshape(3, 1)
    pred_traj = [pos_prev.flatten(), pos_current.flatten()]
    
    for t in range(1, n_timesteps - 1):
        x_in = np.vstack((pos_current, pos_prev))
        y_pred, _ = forward(x_in)
        pred_traj.append(y_pred.flatten())
        pos_prev = pos_current
        pos_current = y_pred
    pred_traj = np.array(pred_traj).T
    rollout_trajs.append(pred_traj)
    error = np.sum((pred_traj - test_data[traj, :, :pred_traj.shape[1]])**2) / n_timesteps
    rollout_errors.append(error)

rollout_error_avg = np.mean(rollout_errors)
rollout_error_max = np.max(rollout_errors)
print(f"Roll-Out Error (MSE) mean: {rollout_error_avg:.6f}")
print(f"Roll-Out Error (MSE) max: {rollout_error_max:.6f}")


# Loss
plt.figure(figsize=(8, 5))
plt.plot(loss_history_train, label="Training Loss", color="purple")
plt.plot(loss_history_test, label="Test Loss", color="orange")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs. Epochs")
plt.legend()


# Trajectory
traj_idx = np.argmin(rollout_errors)
true_traj = test_data[traj_idx, :, :]
one_step_traj = one_step_preds[traj_idx]
rollout_traj = rollout_trajs[traj_idx]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(true_traj[0, :], true_traj[1, :], true_traj[2, :], label="True Trajectory", color="blue", linewidth=2)
ax.plot(one_step_traj[0, :], one_step_traj[1, :], one_step_traj[2, :], label="One-Step Prediction", color="red", linestyle="--", linewidth=2)
ax.plot(rollout_traj[0, :], rollout_traj[1, :], rollout_traj[2, :], label="Roll-Out Prediction", color="green", linestyle=":", linewidth=2)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Trajectory: True vs. NN Predictions")
ax.legend()
plt.show()