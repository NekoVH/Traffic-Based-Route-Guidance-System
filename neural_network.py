import numpy as np
from dataclasses import dataclass

def activation_function(activation_type, values):
	match activation_type:
		case "sigmoid":
			return 1 / (1 + np.exp(-values))
		case "relu":
			return np.maximum(0, values)

@dataclass
class DataPoint:
	inputs: np.ndarray
	expected_outputs: np.ndarray
	label: int

	def one_hot_encoding(self):
		one_hot = np.zeros(len(inputs))
		one_hot[label] = 1
		return one_hot

class Layer():
	def __init__(self, num_nodes_in: int, num_nodes_out: int, activation_type: str):
		self.num_nodes_in = num_nodes_in
		self.num_nodes_out = num_nodes_out
		self.activation_type = activation_type

		self.weights = np.random.rand(num_nodes_in, num_nodes_out)
		self.biases = np.random.rand(num_nodes_out)

	def feed_forward(self, inputs: list) -> np.ndarray:
		activations = np.zeros((self.num_nodes_out,))

		for node_out in range(0, self.num_nodes_out):
			weighted_input = self.biases[node_out]
			for node_in in range(0, self.num_nodes_in):
				weighted_input += inputs[node_in] * self.weights[node_in, node_out]
			activations[node_out] = weighted_input

		activations = activation_function(self.activation_type, activations)
		return activations

class NeuralNetwork():
	def __init__(self, layer_sizes: list, activation_types: str):
		self.layers = []
		for i in range(0, len(layer_sizes) - 1):
			self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], activation_types[i]))

	def feed_forward(self, inputs: list):
		for layer in self.layers:
			inputs = layer.feed_forward(inputs)
		return inputs
