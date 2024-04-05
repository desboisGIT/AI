import math
import random
import pygame

#This is my first step with AI, i used 3Blue1Brown Video for the theory, i hope i will continue this way, (update test)

pygame.init()

background_colour = (255, 255, 255)
(width, height) = (1200, 1000)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Neural Network Display')
screen.fill(background_colour)

ACTIVATION = 1
WEIGHT = 0


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def draw_neuron(x, y, activation, radius):
    color = (0, activation*255, 0)  
    pygame.draw.circle(screen, color, (x, y), radius)
    font = pygame.font.SysFont(None, 24)
    text = font.render(str(round(activation, 2)), True, (255, 0, 0))
    screen.blit(text, (x - 12, y - 32))


def draw_connection(x1, y1, x2, y2):
    pygame.draw.line(screen, (0, 0, 0), (x1, y1), (x2, y2), 2)


def draw_network(layers, radius):
    layer_spacing = width / (len(layers) + 1)
    neuron_spacing = height / (max(len(layer) for layer in layers) + 1)
    
    for i, layer in enumerate(layers):
        x = (i + 1) * layer_spacing
        for j, neuron in enumerate(layer):
            y = (j + 1) * neuron_spacing
            draw_neuron(int(x), int(y), neuron[ACTIVATION],radius)
            if i > 0:
                for k, prev_neuron in enumerate(layers[i - 1]):
                    prev_y = (k + 1) * neuron_spacing
                    draw_connection(int(x), int(y), int(x - layer_spacing), int(prev_y))


def calculate_neuron_activation(neuron_layer, bias):
    weighted_sum = 0
    try:
        for neuron in layers[neuron_layer-1]:
            weighted_sum += neuron[ACTIVATION] * neuron[WEIGHT]
        return sigmoid(weighted_sum - bias)


    except:
       print("does neuron_layer-1 exist?")

bias = 2

pixels = [1,1,1,
          0,0,1,
          0,1,1,
          0,0,1,
          1,1,1] # 3 

# 1 = active pixel, 0 = not active


HIDDEN_LAYER_SIZE = 12
OUTPUT_LAYER_SIZE = 10


INPUT_LAYER = 0
HIDDEN_LAYER = 1
OUTPUT_LAYER = 2

layers = [
    [[random.randint(-2, 2), a] for a in pixels],                    # Input layer
    [[random.randint(-2, 2), 0] for _ in range(HIDDEN_LAYER_SIZE)],  # Hidden layer
    [[random.randint(-2, 2), 0] for _ in range(OUTPUT_LAYER_SIZE)],  # Output layer
]

for neuron_index, neuron in enumerate(layers[INPUT_LAYER]):
   print("LAYER: 1,  Neuron "+str(neuron_index+1)+", activation= " + str(neuron[ACTIVATION])+"   weight= "+str(neuron[WEIGHT])+"\n")


for neuron_index, neuron in enumerate(layers[HIDDEN_LAYER]):
    neuron[ACTIVATION] = calculate_neuron_activation(HIDDEN_LAYER, bias)

    print("LAYER: 2,  Neuron "+str(neuron_index+1)+", activation= " + str(neuron[ACTIVATION])+"   weight= "+str(neuron[WEIGHT])+"\n")


for neuron_index, neuron in enumerate(layers[OUTPUT_LAYER]):
    neuron[ACTIVATION] = calculate_neuron_activation(OUTPUT_LAYER, bias)

    print("LAYER: 3,  Neuron "+str(neuron_index+1)+", activation= " + str(neuron[ACTIVATION])+"   weight= "+str(neuron[WEIGHT])+"\n")


screen.fill(background_colour)
draw_network(layers, 7)

pygame.display.flip()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    #screen.fill(background_colour)
    #draw_network(layers, 7)
    pygame.display.flip()

pygame.quit()