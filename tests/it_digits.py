import unittest
import numpy as np
from nn import network as net, optimizers as opt, lr, tester, trainer, reporter, functions as funcs




class LearnToRecognizeDigits(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        self.reporter = reporter.Reporter()
        self.trainer = trainer.Trainer(train_inputs, train_outputs, batch_size=3)
        self.tester = tester.Tester(test_inputs, test_outputs)

    def test_GD(self):
        cost = funcs.quadratic_cost
        nn = net.NeuralNetwork(layers=(64, 64, 10), activation=(funcs.relu, funcs.sigmoid))
        optimizer = opt.GradientDescent(lr=lr.Fixed(0.5), cost=cost).initialize(nn)

        for e in range(5):
            self.reporter.report_train(e, self.trainer.train_epoch(nn, optimizer))

        test = self.reporter.report_test(self.tester.test(nn, cost))

        self.assertGreaterEqual(test.accuracy(), 0.9, "Test accuracy too low")

    def test_GD_with_momemtum(self):
        cost = funcs.quadratic_cost
        nn = net.NeuralNetwork(layers=(64, 64, 10), activation=(funcs.relu, funcs.sigmoid))
        optimizer = opt.GradientDescentWithMomentum(lr=lr.Fixed(0.5), cost=cost).initialize(nn)

        for e in range(5):
            self.reporter.report_train(e, self.trainer.train_epoch(nn, optimizer))

        test = self.reporter.report_test(self.tester.test(nn, cost))

        self.assertGreaterEqual(test.accuracy(), 0.9, "Test accuracy too low")

    def test_RMSProd(self):
        cost = funcs.quadratic_cost
        nn = net.NeuralNetwork(layers=(64, 64, 10), activation=(funcs.relu, funcs.sigmoid))
        optimizer = opt.RMSProp(lr=lr.Fixed(0.01), cost=cost).initialize(nn)

        for e in range(5):
            self.reporter.report_train(e, self.trainer.train_epoch(nn, optimizer))

        test = self.reporter.report_test(self.tester.test(nn, cost))

        self.assertGreaterEqual(test.accuracy(), 0.91, "Test accuracy too low")

    def test_adam(self):
        cost = funcs.quadratic_cost
        nn = net.NeuralNetwork(layers=(64, 64, 10), activation=(funcs.relu, funcs.sigmoid))
        optimizer = opt.Adam(lr=lr.Fixed(0.005), cost=cost).initialize(nn)

        for e in range(5):
            self.reporter.report_train(e, self.trainer.train_epoch(nn, optimizer))

        test = self.reporter.report_test(self.tester.test(nn, cost))

        self.assertGreaterEqual(test.accuracy(), 0.9, "Test accuracy too low")

    def test_adam2(self):
        cost = funcs.quadratic_cost
        nn = net.NeuralNetwork(layers=(64, 64, 64, 10), activation=(funcs.relu, funcs.relu, funcs.sigmoid),
                               w_initializers=(net.relu_xavier, net.relu_xavier, net.random_weights))
        optimizer = opt.Adam(lr=lr.Fixed(0.0009), cost=cost).initialize(nn)

        for e in range(5):
            self.reporter.report_train(e, self.trainer.train_epoch(nn, optimizer))

        test = self.reporter.report_test(self.tester.test(nn, cost))

        self.assertGreaterEqual(test.accuracy(), 0.9, "Test accuracy too low")

    def test_adam3(self):
        cost = funcs.quadratic_cost
        nn = net.NeuralNetwork(layers=(64, 64, 64, 64, 10),
                               activation=(funcs.relu, funcs.relu, funcs.relu, funcs.sigmoid))
        optimizer = opt.Adam(lr=lr.Fixed(0.000095), cost=cost).initialize(nn)

        for e in range(50):
            self.reporter.report_train(e, self.trainer.train_epoch(nn, optimizer))

        test = self.reporter.report_test(self.tester.test(nn, cost))

        self.assertGreaterEqual(test.accuracy(), 0.89, "Test accuracy too low")


# 8x8=64
def get_dataset(filename):
    lines = [[int(value) for value in line.split(",")]
             for line in open(filename).readlines()]
    return [get_desired_output(line[64]) for line in lines], [np.array([value / 16 for value in line[0:64]]) for line in
                                                              lines]


# returns all the network output for a given number.
def get_desired_output(number):
    desired_output = np.zeros(10, dtype=np.float64)
    desired_output[number] = 1
    return desired_output


(test_outputs, test_inputs) = get_dataset("dataset/optdigits.tes")
(train_outputs, train_inputs) = get_dataset("dataset/optdigits.tra")
