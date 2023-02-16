import numpy as np


class Tester:

    def __init__(self, inputs, desired_outputs):
        self.inputs = inputs
        self.desired_outputs = desired_outputs
        self.correct = np.zeros(len(inputs), dtype=np.bool)
        self.costs = np.zeros(len(inputs))

    def test(self, nn, cost):
        return Test(self.inputs, self.desired_outputs).test(nn, cost)


class Test:

    def __init__(self, inputs, desired_outputs):
        self.inputs = inputs
        self.desired_outputs = desired_outputs
        self.correct = np.zeros(len(inputs), dtype=np.bool)
        self.costs = np.zeros(len(inputs))

    def test(self, nn, cost):
        for i, (input_, desired_output) in enumerate(zip(self.inputs, self.desired_outputs)):
            output = nn.run(input_)
            self.correct[i] = nn.activation[-1].is_output_equal(output, desired_output)
            self.costs = cost.calc(output, desired_output)
        return self

    def accuracy(self):
        return self.correct.sum() / len(self.correct)

    def average_cost(self):
        return np.average(self.costs)

    def max_cost(self):
        return np.max(self.costs)

    def min_cost(self):
        return np.min(self.costs)
