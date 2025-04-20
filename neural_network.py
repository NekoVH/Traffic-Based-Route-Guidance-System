import numpy as np

def activation_function(activation_type, value):
	match activation_type:
		case "sigmoid":
			return 1 / (1 + np.exp(-value))
		case "relu":
			return max(0, value)

class Layer():
	def __init__(self, num_nodes_in, num_nodes_out, activation_type):
		self.num_nodes_in = num_nodes_in
		self.num_nodes_out = num_nodes_out
		self.activation_type = activation_type

		self.weights = np.ones((num_nodes_in, num_nodes_out))
		self.biases = np.ones((num_nodes_out,))

	def calculate_outputs(self, inputs):
		activations = np.zeros((self.num_nodes_out,))

		for node_out in range(0, self.num_nodes_out):
			weighted_input = self.biases[node_out]
			for node_in in range(0, self.num_nodes_in):
				weighted_input += inputs[node_in] * self.weights[node_in, node_out]
			activations[node_out] = activation_function(self.activation_type,weighted_input)

		return activations


class NeuralNetwork():
	def __init__(self, layer_sizes, activation_type="sigmoid"):
		self.layers = []
		for i in range(0, len(layer_sizes) - 1):
			self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], activation_type))

	def calculate_outputs(self, inputs):
		for layer in self.layers:
			inputs = layer.calculate_outputs(inputs)
		return inputs
