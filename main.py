import numpy as np

def tanh_derivative(x):
    return 1.0 - np.tanh(x) ** 2

class NeuralMath:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        current_input = X

        for i in range(self.num_layers - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)

            if i == self.num_layers - 2:
                current_input = z
            else:
                current_input = np.tanh(z)

            self.activations.append(current_input)
        return current_input
    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_true, learning_rate=0.01):
        m = y_true.shape[0]

        grad = 2 * (self.activations[-1] - y_true) / m
        for i in range(self.num_layers - 2, -1, -1):
            if i < self.num_layers - 2:
                grad = grad * tanh_derivative(self.z_values[i])

            dw = np.dot(self.activations[i].T, grad)
            db = np.sum(grad, axis=0, keepdims=True)

            if i > 0:
                grad = np.dot(grad, self.weights[i].T)

            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

if __name__ == "__main__":
    print("initializing the neural calculator ..")

    X_raw= np.array([[1.0], [2.0], [3.0], [4.0]])
    y_raw = np.array([[2.0], [4.0], [6.0], [8.0]])

    scale_factor = 10.0
    X_train = X_raw / scale_factor
    y_train = y_raw / scale_factor

    nn = NeuralMath([1, 8, 1])
    print("intializing the trainment (500 cycles)")

    for epoch in range(3000):
        y_pred = nn.forward(X_train)

        loss = nn.compute_loss(y_pred, y_train)
        nn.backward(y_train, learning_rate=0.05)

        if epoch % 500 == 0:
            print(f"Cycle {epoch:3d} | Loss: {loss:.6f}")

    print("\nTraining Concluded with Success")
    print("-- Testing the Neural Network Trained --")

    rawtest_x = np.array([[5.0], [10.0], [15.0]])
    normtest_x = rawtest_x / scale_factor

    norm_pred = nn.forward(normtest_x)
    pred_real = norm_pred * scale_factor

    for i in range(len(rawtest_x)):
        entry = rawtest_x[i][0]
        waited = entry * 2.0
        predicted = pred_real[i][0]

        print(f"Entry: {entry:4.1f} -> Predicted: {predicted:6.2f} (waited: {waited:5.1f})")
