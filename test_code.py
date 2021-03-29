import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = [[4.9, 3.,  1.4, 0.2]]
weights_1 = np.random.normal(scale=0.5, size=(4, 2))
weights_2 = np.random.normal(scale=0.5, size=(2, 3))
bias_hidden = np.ones((2, 1))

hidden_layer_inputs = np.dot(x, weights_1) + bias_hidden
hidden_layer_outputs = sigmoid(hidden_layer_inputs)

output_layer_inputs = np.dot(hidden_layer_outputs, weights_2) + bias_hidden
output_layer_outputs = sigmoid(output_layer_inputs)

print(output_layer_inputs)
print(output_layer_outputs)
