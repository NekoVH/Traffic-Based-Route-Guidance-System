from neural_network import *

network = NeuralNetwork([2,3,2], ["relu", "sigmoid"])

inputs = [0.5, 0.6]

print(f"Inputs: {inputs}\n")
for layer in network.layers:
	print(f"Weights: {layer.weights}")
	print(f"Biases: {layer.biases}")
	print(f"Activation type: {layer.activation_type}\n")
print(f"Output of the network is: {network.feed_forward(inputs)}")
