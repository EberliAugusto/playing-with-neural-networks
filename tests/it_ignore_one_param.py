import unittest
import numpy as np
from nn import network as net, optimizers as opt, tester, trainer, reporter, functions as funcs



class LearnToIgnoreOneParameter(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.trainer = trainer.Trainer(inputs, outputs, batch_size=1)
        self.tester = tester.Tester(inputs, outputs)
        self.reporter = reporter.Reporter()

    def test_gradient_descent(self):
        cost = funcs.quadratic_cost
        nn = net.NeuralNetwork((2, 2, 1), activation=[funcs.relu, funcs.sigmoid])
        d = opt.GradientDescent(0.5, cost=cost).initialize(nn)

        for e in range(10):
            self.reporter.report_train(e, self.trainer.train_epoch(nn, d))

        test = self.reporter.report_test(self.tester.test(nn, cost))

        self.assertEqual(1., test.accuracy())

    def test_gradient_descent1(self):
        cost = funcs.quadratic_cost
        nn = net.NeuralNetwork((2, 1), activation=[funcs.sigmoid])
        d = opt.GradientDescent(0.5, cost=cost).initialize(nn)

        for e in range(5):
            self.reporter.report_train(e, self.trainer.train_epoch(nn, d))

        test = self.reporter.report_test(self.tester.test(nn, cost))

        self.assertEqual(1., test.accuracy())

    def test_gradient_descent_with_momemtum(self):
        cost = funcs.quadratic_cost
        nn = net.NeuralNetwork((2, 2, 1), activation=[funcs.relu, funcs.sigmoid])
        d = opt.GradientDescentWithMomentum(0.5, cost=cost).initialize(nn)

        for e in range(16):
            self.reporter.report_train(e, self.trainer.train_epoch(nn, d))

        test = self.reporter.report_test(self.tester.test(nn, cost))

        self.assertEqual(1., test.accuracy())


# nn = net.NeuralNet((2, 1), activation=(net.sigmoid), optimizer=net.GradientDescent(0.5))


inputs = [
    [0.5, 1],
    [0.5, 0]]

outputs = [[1],
           [0]]

if __name__ == '__main__':
    unittest.main()
