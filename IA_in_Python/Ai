import math
import random

sigmoid = lambda x: 1 / (1 + math.exp(-x))

def createNeuralNetwork(num_input_neurons, num_hidden_neurons, num_output_neurons, num_hidden_layers):
    neural_network = []
    
    # Input layer
    input_layer = [[] for _ in range(num_input_neurons)]
    neural_network.append(input_layer)

    # Hidden layers
    for _ in range(num_hidden_layers):
        hidden_layer = [[] for _ in range(num_hidden_neurons)]
        neural_network.append(hidden_layer)
    
    # Output layer
    output_layer = [[] for _ in range(num_output_neurons)]
    neural_network.append(output_layer)

    return neural_network

def setRandomNeuron(neural_network):
    updated_network = neural_network.copy()
    for layerIndex, layer in enumerate(updated_network):
        for neuronIndex, neuron in enumerate(layer):
            neuron.append(random.uniform(0, 1))  # activation
            if layerIndex > 0:  # Exclude input layer
                weights = [random.uniform(-1, 1) for _ in range(len(updated_network[layerIndex - 1]))]
                neuron.append(weights)  # weights
            neuron.append(random.uniform(-1, 1))  # bias
    return updated_network

def showNetwork(neural_network):
    for i, layer in enumerate(neural_network):
        print("layer ",i, ": ")
        for neuronIndex, neuron in enumerate(layer):
            print(" neuron ", neuronIndex,":")
            try:
                print("     activation : ", neuron[0])
            except:
                print("     activation value not set")
            try:
                print("     weights : ", neuron[1])
            except:
                print("     weights value not set")
            try:    
                print("     bias : ", neuron[2])
            except:
                print("     bias value not set")
        print("\n")

def forwardPropagation(neural_network, inputs):
    # Set input values
    for neuron, input_val in zip(neural_network[0], inputs):
        neuron[0] = input_val  # Update activation value in input layer

    
    for layerIndex, layer in enumerate(neural_network[1:]):
        prev_layer = neural_network[layerIndex]  
        for neuron in layer:
            weighted_sum = 0
            
            for prev_neuron, weight in zip(prev_layer, neuron[1]):
                weighted_sum += prev_neuron[0] * weight
            weighted_sum += neuron[2]  # Add bias
            neuron[0] = sigmoid(weighted_sum)  # apply activation function

    # Return output values
    return [neuron[0] for neuron in neural_network[-1]]


neural_network = createNeuralNetwork(2, 2, 1, 1)
neural_network = setRandomNeuron(neural_network)

inputs = [0.5, 0.3]

outputs = forwardPropagation(neural_network, inputs)
print("Output:", outputs)

showNetwork(neural_network)
