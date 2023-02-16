import numpy as np


#rectified linear unit
class __ReLU:

    def __init__(self):
        pass

    @staticmethod
    def calc(x):
        return np.maximum(x, 0.)

    @staticmethod
    def prime(x):
        return np.where(x > 0., 1., 0.)


relu = __ReLU()


class __Sigmoid:

    def __init__(self):
        pass

    def calc(self, x):
        return 1. / (1. + np.e ** -x)

    def prime(self, x):
        sig = self.calc(x)
        return sig * (1. - sig)

    @staticmethod
    def is_output_equal(actual, desired):
        return np.array_equal(np.round(actual), desired)


sigmoid = __Sigmoid()


class __QuadraticCost:

    def __init__(self):
        pass

    @staticmethod
    def calc(actual, expected):
        return np.sum((actual - expected) ** 2.)

    @staticmethod
    def prime(actual, expected):
        return (actual - expected) * 2.


quadratic_cost = __QuadraticCost()
