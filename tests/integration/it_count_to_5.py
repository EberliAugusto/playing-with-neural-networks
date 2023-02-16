import unittest
import numpy as np
from nn import network as net, optimizers as opt, lr, tester, trainer, reporter, functions as funcs



class LearnToCountToFive(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        self.reporter = reporter.Reporter()
        self.trainer = trainer.Trainer(input, output, batch_size=2)
        self.tester = tester.Tester(input, output)

    def test_adam(self):
        cost = funcs.quadratic_cost
        nn = net.NeuralNetwork((5, 7, 5), activation=(funcs.relu, funcs.sigmoid))
        optimizer = opt.GradientDescent(lr.Fixed(0.5), cost=cost).initialize(nn)

        for e in range(100):
            self.reporter.report_train(e, self.trainer.train_epoch(nn, optimizer))

        test = self.reporter.report_test(self.tester.test(nn,cost=cost))
        self.assertGreater(test.accuracy(), 0.9)


input = np.zeros((5, 5))
output = np.zeros((5, 5))
for i in range(5):
    output[i, i] = 1
    input[i, i] = 1
