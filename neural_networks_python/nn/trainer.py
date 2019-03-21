from concurrent.futures import ThreadPoolExecutor
from nn import utils

import numpy as np

class Trainer:

    def __init__(self, inputs, outputs, batch_size=32):
        self.inputs = inputs
        self.outputs = outputs
        self.batch_size = batch_size
        self.count = None
        self.averages = utils.Averages(1, len(inputs))
        self.accuracy = utils.Percentage(1,len(inputs))


    def train_epoch(self, nn, opt):
        self.averages.reset()
        self.accuracy.reset()
        if self.batch_size > 1:
            self.__train_batches(nn, opt)
        else:
            self.__train_sequential(nn, opt)
        return self

    def __train_sequential(self, nn, opt):
        for desired_output, input_ in zip(self.outputs, self.inputs):
            z_by_layer, a_by_layer = nn.run_complete(input_)
            cost_prime_on_w, cost_prime_on_b = opt.calculate_gradients(nn, z_by_layer, a_by_layer, input_,
                                                                       desired_output)
            opt.update_network(nn, cost_prime_on_w, cost_prime_on_b)
            self.__update_stats(a_by_layer, desired_output, nn, opt)

    def __update_stats(self, a_by_layer, desired_output, nn, opt):
        output = a_by_layer[-1]
        self.averages.add(opt.cost.calc(output, desired_output))
        self.accuracy.add(nn.activation[-1].is_output_equal(output, desired_output))

    def __train_batches(self, nn, opt):
        for batch_inputs, batch_outputs in iterate_batch(self.inputs, self.outputs, self.batch_size):
            self.__train_mini_batch(nn, opt, batch_inputs, batch_outputs)

    def __train_mini_batch(self, nn, opt, batch_inputs, batch_outputs):
        results = self.__run_bach(nn, batch_inputs)

        self.__update_stats_batch(batch_inputs, batch_outputs, nn, opt, results)
        gradients = self.__calculate_gradients(opt, batch_inputs, batch_outputs, nn, results)
        costs_prime_on_w, costs_prime_on_b = self.__avg_gradients(nn, gradients)
        opt.update_network(nn, costs_prime_on_w, costs_prime_on_b)

    def __update_stats_batch(self, batch_inputs, batch_outputs, nn, opt, results):
        for input_, desired_output, (_, a_by_layer) in zip(batch_inputs, batch_outputs, results):
            self.__update_stats(a_by_layer, desired_output, nn, opt)

    @staticmethod
    def __calculate_gradients(opt, batch_inputs, batch_outputs, nn, results):
        return [opt.calculate_gradients(nn, z_by_layer, a_by_layer, input_, desired_output)
                for input_, desired_output, (z_by_layer, a_by_layer) in
                zip(batch_inputs, batch_outputs, results)]

    @staticmethod
    def __run_bach(nn, inputs):
        return [nn.run_complete(input_) for input_ in inputs]

    @staticmethod
    def __avg_gradients(nn, gradients):
        primes_on_w = [cost_prime_on_w for cost_prime_on_w, _ in gradients]
        primes_on_b = [cost_prime_on_b for _, cost_prime_on_b in gradients]

        # TODO do better with numpy.
        primes_on_b_t = [[p[l] for p in primes_on_b] for l in range(nn.depth)]
        primes_on_w_t = [[p[l] for p in primes_on_w] for l in range(nn.depth)]
        #
        avg_prime_on_w = [np.average(primes_on_w_in_l, axis=0) for primes_on_w_in_l in primes_on_w_t]
        avg_prime_on_b = [np.average(primes_on_b_in_l, axis=0) for primes_on_b_in_l in primes_on_b_t]

        return avg_prime_on_w, avg_prime_on_b


def iterate_batch(input, output, batch_size):
    counter = 0
    while counter < len(input):
        yield input[counter: counter + batch_size], output[counter: counter + batch_size]
        counter += batch_size
