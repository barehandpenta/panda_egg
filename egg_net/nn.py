import numpy as np
import scipy.special
import csv

class NeuralNetwork:

    def __init__(self, INodes, HLayers, HNodes, ONodes, learning_rate = 0.01):

        self.INodes = INodes
        self.HLayers = HLayers
        self.HNodes = HNodes
        self.ONodes = ONodes
        self.lr = learning_rate

        self.input_weight = np.random.random_sample((self.HNodes, self.INodes)) - 0.5
        self.hidden_weight = np.random.random_sample((self.HLayers - 1, self.HNodes, self.HNodes)) - 0.5
        self.output_weight = np.random.random_sample((self.ONodes, self.HNodes)) - 0.5

        self.input_bias = np.random.random_sample((self.HNodes, 1)) - 0.5
        self.hidden_bias = np.random.random_sample((self.HLayers - 1, self.HNodes, 1)) - 0.5
        self.output_bias = np.random.random_sample((self.ONodes, 1)) - 0.5

        self.weight_list = []
        self.bias_list = []
        self.activation_list = []
        self.delta_weight = []
        self.delta_bias = []


        for i in range(HLayers + 2):
            if i == 0:
                self.weight_list.append([0, 0, 0])
            elif i == 1:
                self.weight_list.append(self.input_weight)
            elif (i > 1) & (i < HLayers + 1):
                self.weight_list.append(self.hidden_weight[i - 2])
            elif i == HLayers + 1:
                self.weight_list.append(self.output_weight)

        for i in range(HLayers + 2):
            if i == 0:
                self.bias_list.append([0, 0, 0])
            elif i == 1:
                self.bias_list.append(self.input_bias)
            elif (i > 1) & (i < HLayers + 1):
                self.bias_list.append(self.hidden_bias[i - 2])
            elif i == HLayers + 1:
                self.bias_list.append(self.output_bias)

        for i in range(HLayers + 2):
            self.activation_list.append([])
        for i in range(HLayers + 2):
            if i == 0:
                self.delta_weight.append(['null'])
                self.delta_bias.append(['null'])
            else:
                self.delta_weight.append([])
                self.delta_bias.append([])
    def train(self, inputs, targets):
        targets = np.array(targets, ndmin=2).T
        guess = self.feedFoward(inputs)
        err = guess - targets
        D = []

        for i in range(self.HLayers + 2):
            D.append([])
        # Backpropagation:
        # Calculate Layer errors:
        for i in range(self.HLayers + 1, 0, -1):
            if i == self.HLayers + 1:
                D[i] = err*self.activation_list[i] * (1 - self.activation_list[i])
            else:
                D[i] = np.dot(self.weight_list[i+1].T, D[i+1])*self.activation_list[i] * (1 - self.activation_list[i])
        # Calculate Partial derivative
        for i in range(self.HLayers + 1, 0, -1):
            self.delta_weight[i] = np.dot(D[i], self.activation_list[i-1].T)
            self.delta_bias[i] = D[i]

        for i in range(self.HLayers + 1, 0, -1):
            self.weight_list[i] -= self.lr*self.delta_weight[i]
            self.bias_list[i] -= self.lr*self.delta_bias[i]
        return np.sum(err*err)/3

    def activation(self, x):
        return scipy.special.expit(x)
    def feedFoward(self, inputs):
        inputs = np.array(inputs, ndmin=2).T
        self.activation_list[0] = inputs
        for i in range(1, self.HLayers + 2):
            self.activation_list[i] = self.activation(np.dot(self.weight_list[i], self.activation_list[i-1]) + self.bias_list[i])
        return self.activation_list[self.HLayers + 1]

    def save_model(self, filename):
        model_file = open(filename, 'w', newline='')
        writer = csv.writer(model_file)
        for i in range(self.HLayers + 2):
            weights = np.asarray(self.weight_list[i])
            newList = weights.flatten()
            writer.writerow(newList)
        for i in range(self.HLayers + 2):
            biases = np.asarray(self.bias_list[i])
            newList = biases.flatten()
            writer.writerow(newList)

    def load_model(self, weights_file):
        params = open(weights_file, 'r').readlines()
        for i in range(len(params)):
            params[i] = params[i].split(',')
        # Load weights into the network
        for i in range(1, int(len(params)/2)):
            shape = np.asfarray(self.weight_list[i]).shape
            w = np.asfarray(params[i])
            self.weight_list[i] = w.reshape(shape)
        # Load biases into the network:
        for i in range(1, int(len(params) / 2)):
            shape = np.asfarray(self.bias_list[i]).shape
            b = np.asfarray(params[i + int(len(params) / 2)])
            self.bias_list[i] = b.reshape(shape)













