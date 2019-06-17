from egg_net.nn import NeuralNetwork
from common.common import open_csv_file, preProcessing


inputs, targets = preProcessing('konel_egg_train.csv')
network = NeuralNetwork(34, 2, 15, 3, 0.01)
for e in range(1000):
    for i in range(len(targets)):
        cost = network.train(inputs[i], targets[i])
network.save_model('../data/egg_model_weights.csv')
