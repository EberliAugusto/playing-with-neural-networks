import unittest
import numpy as np
from nn import network as net, optimizers as opt, lr, tester, trainer, reporter, functions as funcs



class LearnBooleanOperations(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        self.trainer = trainer.Trainer(input_, output, batch_size=1)
        self.tester = tester.Tester(input_, output)
        self.reporter = reporter.Reporter()

    def test_gradient_descent(self):
        cost = funcs.quadratic_cost
        nn = net.NeuralNetwork((2, 4, 3), activation=(funcs.relu, funcs.sigmoid))
        optimizer = opt.GradientDescent(lr.Fixed(0.5), cost=cost).initialize(nn)
        for e in range(60):
            self.reporter.report_train(e, self.trainer.train_epoch(nn, optimizer))
        accuracy = self.tester.test(nn, cost).accuracy()
        self.assertEqual(1, accuracy)


input_ = np.array([[1, 1],
                  [1, 0],
                  [0, 1],
                  [0, 0]])

output = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1]])
