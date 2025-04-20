from neural_network import *

network = NeuralNetwork([2,3,2])

print(f"Output of the network is: {network.calculate_outputs([0.8, 0.9])}")
