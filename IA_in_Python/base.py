import matplotlib.pyplot as plt
import math
import random




neuronInInputLayer = 15
neuronInHiddenLayer = 13
neuronInOutputLayer = 10

ACTIVATION = 0
WEIGHT = 1

inputPixel = [ #made with gpt
    # Digit 0
    [
        1, 1, 1,
        1, 0, 1,
        1, 0, 1,
        1, 0, 1,
        1, 1, 1,
    ],
    # Digit 1
    [
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
    ],
    # Digit 2
    [
        1, 1, 1,
        0, 0, 1,
        1, 1, 1,
        1, 0, 0,
        1, 1, 1,
    ],
    # Digit 3
    [
        1, 1, 1,
        0, 0, 1,
        1, 1, 1,
        0, 0, 1,
        1, 1, 1,
    ],
    # Digit 4
    [
        1, 0, 1,
        1, 0, 1,
        1, 1, 1,
        0, 0, 1,
        0, 0, 1,
    ],
    # Digit 5
    [
        1, 1, 1,
        1, 0, 0,
        1, 1, 1,
        0, 0, 1,
        1, 1, 1,
    ],
    # Digit 6
    [
        1, 1, 1,
        1, 0, 0,
        1, 1, 1,
        1, 0, 1,
        1, 1, 1,
    ],
    # Digit 7
    [
        1, 1, 1,
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
    ],
    # Digit 8
    [
        1, 1, 1,
        1, 0, 1,
        1, 1, 1,
        1, 0, 1,
        1, 1, 1,
    ],
    # Digit 9
    [
        1, 1, 1,
        1, 0, 1,
        1, 1, 1,
        0, 0, 1,
        1, 1, 1,
    ],
]
correctAnswer = [ #made with gpt
    # Digit 0
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # Digit 1
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    # Digit 2
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    # Digit 3
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    # Digit 4
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    # Digit 5
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    # Digit 6
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    # Digit 7
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    # Digit 8
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    # Digit 9
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
]
choosenNumber = random.randint(0,9)

sigmoid = lambda x: 1 / (1 + math.exp(-x))

def setRandomWeight(network):
    for layerIndex, layer in enumerate(network[:-1]):
        for neuron in layer:
            neuron[1] = [random.randint(-2, 2) for _ in range(len(network[layerIndex + 1]))]

def calculateNeuralActivation(layers, neuronIndex, layerIndex):
    activation = 0
    for neuron in layers[layerIndex-1]:
        activation += neuron[0] * neuron[1][neuronIndex]
    layers[layerIndex][neuronIndex][0] = sigmoid(activation)

def solveNeuralNetwork(network):
    for i in range(1,len(network)):
        for neuronIndex in range(len(network[i])):
            calculateNeuralActivation(network, neuronIndex, i)

def calculateOutputAccuracy(outputLayer):
    accuracy = 0
    for neuronIndex, neuron in enumerate(outputLayer):
        accuracy += (neuron[ACTIVATION]-correctAnswer[choosenNumber][neuronIndex])**2
    return accuracy


print(choosenNumber)
layers = [
    [[inputPixel[choosenNumber][neuron], [random.randint(-2, 2) for _ in range(neuronInHiddenLayer)]] for neuron in range(neuronInInputLayer)],
    [[0, [random.randint(-2, 2) for _ in range(neuronInOutputLayer)]] for _ in range(neuronInHiddenLayer)],
    [[0, []] for _ in range(neuronInOutputLayer)]
]


setRandomWeight(layers)


solveNeuralNetwork(layers)

print(layers[2])
print(calculateOutputAccuracy(layers[2]))


#rendering stuff
plt.figure(figsize=(17, 6))
for layerIndex, layer in enumerate(layers):
    print(layerIndex)
    for neuronIndex, neuron in enumerate(layer):
        x = layerIndex + 1
        y = neuronIndex
        activation = neuron[0]
        plt.scatter(x, y, s=100, color='green', alpha=activation)  # neron acti
        plt.text(x, y+0.35, f'{activation:.2f}', ha='center', va='center',fontsize=16)
        if layerIndex==2:
            plt.text(x+0.07, y, f'{neuronIndex}', ha='center', va='center',fontsize=16)

        if layerIndex < len(layers) - 1:
            next_layer = layers[layerIndex + 1]
            for next_neuronIndex, next_neuron in enumerate(next_layer):
                next_x = layerIndex + 2
                next_y = next_neuronIndex
                weight = neuron[1][next_neuronIndex]
                plt.plot([x, next_x], [y, next_y], color='red', alpha=abs(weight/2), linewidth=2)  # connection
plt.text(1,15, "number:" + str(choosenNumber) + "  accuracy: "+ str(calculateOutputAccuracy(layers[2])), ha='center', va='center',fontsize=16)

plt.xlabel('Layer')
plt.ylabel('Neuron')
plt.title('Neural Network Visualization By Robin')
plt.grid(False)
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().autoscale(enable=True, axis='both')
plt.gca().set_aspect('auto')
plt.show()

