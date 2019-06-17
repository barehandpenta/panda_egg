import numpy as np
from egg_net.nn import NeuralNetwork
from common.common import preProcessing

inputs, targets = preProcessing('konel_egg_test.csv')

network = NeuralNetwork(34, 2, 15, 3)
network.load_model('../data/egg_model_weights.csv')
guess = [None]*len(inputs)
score = 0
for i in range(len(inputs)):
    result = network.feedFoward(inputs[i]).T
    index = np.where(result == np.amax(result))
    result = np.zeros(3) + 0.01
    result[index[1]] = 0.99
    guess[i] = result
for i in range(len(targets)):
    if np.array_equal(guess[i], targets[i]):
        score += 1

print(score)
print(score*100/len(targets))


