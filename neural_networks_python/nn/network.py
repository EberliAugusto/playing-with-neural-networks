from concurrent.futures import ThreadPoolExecutor

import numpy as np

import nn.functions as funcs
from numba.types import *

INPUT = 1
OUTPUT = 0


class ReLUXavier:

    def __init__(self, r=1):
        self.__r = r

    def get_initial_weights(self, nn, l):
        (outputSize, inputSize) = nn.layers[l]
        return np.random.rand(outputSize, inputSize) * np.sqrt(2 / inputSize) * self.__r


relu_xavier = ReLUXavier()


class TanhXavier:

    def __init__(self, r=1):
        self.__r = r

    def get_initial_weights(self, nn, l):
        (outputSize, inputSize) = nn.layers[l]
        return np.random.rand(outputSize, inputSize) * np.sqrt(1 / inputSize) * self.__r


tanh_xavier = TanhXavier()


class RandomWeights:

    def __init__(self, r=0.1):
        self.__r = r

    def get_initial_weights(self, nn, l):
        (outputSize, inputSize) = nn.layers[l]
        return np.random.rand(outputSize, inputSize) * self.__r


random_weights = RandomWeights()


class NeuralNetwork:

    def __init__(self,
                 layers,
                 activation,
                 w_initializers=None):
        self.layers = [[layers[idx], layers[idx - 1]] for idx in range(1, len(layers))]
        self.depth = len(self.layers)
        self.activation = activation
        self.__init_weights(w_initializers)
        self.__init_biases()

    def __init_weights(self, w_initializer):
        if w_initializer is None:
            self.w_by_layer = [random_weights.get_initial_weights(self, l) for l in range(self.depth)]
        else:
            self.w_by_layer = [w_initializer[l].get_initial_weights(self, l) for l in range(self.depth)]

    def __init_biases(self):
        self.b_by_layer = [np.zeros(layer[0]) for layer in self.layers]

    def run(self, inputs):
        return self.run_complete(inputs)[1][-1]

    @staticmethod
    def __z(weights, biases, previous_layer_output):
        return previous_layer_output @ weights.T + biases

    # returns the outputs of all layers, being the output layer on 0.
    def run_complete(self, input_):
        z_by_layer = [None] * self.depth
        a_by_layer = [None] * self.depth

        i = input_
        for l, (w, b) in enumerate(zip(self.w_by_layer, self.b_by_layer)):
            z_by_layer[l] = self.__z(w, b, i)
            a_by_layer[l] = self.activation[l].calc(z_by_layer[l])
            i = a_by_layer[l]
        return z_by_layer, a_by_layer
