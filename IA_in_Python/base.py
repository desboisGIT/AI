import matplotlib.pyplot as plt
import math
import random

neuronInInputLayer = 15
neuronInHiddenLayer = 13
neuronInOutputLayer = 10

ACTIVATION = 0
WEIGHT = 1
BIAS = 2

inputPixel = [
    [
        1, 1, 1,
        1, 0, 1,
        1, 0, 1,
        1, 0, 1,
        1, 1, 1,
    ],
    [
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
    ],
    [
        1, 1, 1,
        0, 0, 1,
        1, 1, 1,
        1, 0, 0,
        1, 1, 1,
    ],
    [
        1, 1, 1,
        0, 0, 1,
        1, 1, 1,
        0, 0, 1,
        1, 1, 1,
    ],
    [
        1, 0, 1,
        1, 0, 1,
        1, 1, 1,
        0, 0, 1,
        0, 0, 1,
    ],
    [
        1, 1, 1,
        1, 0, 0,
        1, 1, 1,
        0, 0, 1,
        1, 1, 1,
    ],
    [
        1, 1, 1,
        1, 0, 0,
        1, 1, 1,
        1, 0, 1,
        1, 1, 1,
    ],
    [
        1, 1, 1,
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
    ],
    [
        1, 1, 1,
        1, 0, 1,
        1, 1, 1,
        1, 0, 1,
        1, 1, 1,
    ],
    [
        1, 1, 1,
        1, 0, 1,
        1, 1, 1,
        0, 0, 1,
        1, 1, 1,
    ],
]

correctAnswer = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
]

sigmoid = lambda x: 1 / (1 + math.exp(-x))

#functions : 

def setRandomWeightAndBias(network):
    for layerIndex, layer in enumerate(network):
        for neuron in layer:
            if layerIndex < 2:
                neuron[1] = [random.randint(-2, 2) for _ in range(len(network[layerIndex + 1]))]
            if layerIndex > 0:
                neuron[2] = random.uniform(-1,1) 

def calculateNeuralActivation(layers, neuronIndex, layerIndex):
    activation = 0
    for neuron in layers[layerIndex-1]:
        activation += neuron[0] * neuron[1][neuronIndex]
    activation += layers[layerIndex][neuronIndex][BIAS]
    layers[layerIndex][neuronIndex][0] = sigmoid(activation)

def solveNeuralNetwork(network):
    for i in range(1, len(network)):
        for neuronIndex in range(len(network[i])):
            calculateNeuralActivation(network, neuronIndex, i)

def calculateOutputAccuracy(outputLayer):
    accuracy = 0
    for neuronIndex, neuron in enumerate(outputLayer):
        accuracy += (neuron[ACTIVATION] - correctAnswer[choosenNumber][neuronIndex])**2
    return accuracy

def makeAccuratySum(iteration_number):
    total_accuracy = 0
    for i in range(iteration_number):
        choosenNumber = iteration_number
        setRandomWeightAndBias(layers)
        solveNeuralNetwork(layers)
        total_accuracy += calculateOutputAccuracy(layers[2])

    average_accuracy = total_accuracy / iteration_number
    return(average_accuracy)

def renderNetwork():
    # Rendering stuff
    plt.figure(figsize=(17, 6))
    for layerIndex, layer in enumerate(layers):
        for neuronIndex, neuron in enumerate(layer):
            x = layerIndex + 1
            y = neuronIndex
            activation = neuron[0]
            plt.scatter(x, y, s=100, color='green', alpha=activation)  # neuron activation
            plt.text(x, y + 0.35, f'{activation:.2f}', ha='center', va='center', fontsize=16)
            if layerIndex == 2:
                plt.text(x + 0.07, y, f'{neuronIndex}', ha='center', va='center', fontsize=16)

            if layerIndex < len(layers) - 1:
                next_layer = layers[layerIndex + 1]
                for next_neuronIndex, next_neuron in enumerate(next_layer):
                    next_x = layerIndex + 2
                    next_y = next_neuronIndex
                    weight = neuron[1][next_neuronIndex]
                    plt.plot([x, next_x], [y, next_y], color='red', alpha=abs(weight / 2), linewidth=abs(weight / 2)+0.5)  # connection

    plt.text(1, 15, "Average accuracy: {:.7f}".format(average_accuracy_precomputed), ha='center', va='center', fontsize=16)
    plt.xlabel('Layer')
    plt.ylabel('Neuron')
    plt.title('Neural Network Visualization By Robin')
    plt.grid(False)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().autoscale(enable=True, axis='both')
    plt.gca().set_aspect('auto')
    plt.show()

choosenNumber = random.randint(0, 9)
layers = [
    [[inputPixel[choosenNumber][neuron], [random.randint(-2, 2) for _ in range(neuronInHiddenLayer)]] for neuron in range(neuronInInputLayer)],
    [[0, [random.randint(-2, 2) for _ in range(neuronInOutputLayer)],0] for _ in range(neuronInHiddenLayer)],
    [[0, [], 0] for _ in range(neuronInOutputLayer)]
]

iteration_nb = 9
average_accuracy_precomputed = makeAccuratySum(iteration_nb)
print("avrage accuracy on:",iteration_nb, " iteration = "+ str(average_accuracy_precomputed))
print("Robin is drawing the graph... please stand by.")
renderNetwork()
