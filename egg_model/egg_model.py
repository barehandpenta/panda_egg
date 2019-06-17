from egg_model.decode import decode_inputs
from egg_net.nn import NeuralNetwork
import numpy as np

class PandaEgg:
    network = NeuralNetwork(34, 2, 15, 3)
    def load_weights(self, weights_file):
        self.network.load_model(weights_file)
    def pose_detect(self, inputs):
        inputs = decode_inputs(inputs)
        guess = self.network.feedFoward(inputs).T
        result = np.where(guess == np.amax(guess))
        return result[1]
