import numpy as np


class EWMA:

    def __init__(self, shape, b, corrected):
        self.b = b
        self.t = 0
        self.corrected = corrected
        self.a = np.zeros(shape)

    def add_values(self, v):
        self.a = self.b * self.a + (1 - self.b) * v
        self.t += 1

    def get(self):
        if self.corrected:
            return self.a / (1 - self.b ** self.t)
        else:
            return self.a


class Percentage:

    def __init__(self, shape, total):
        self.shape = shape
        self.__total = total
        self.__value = 0
        self.__count = 0

    def reset(self):
        self.__value = 0
        self.__count = 0

    def add(self, b):
        #b is boolean
        self.__value += b
        self.__count += 1

    def get_value(self):
        if self.__count != self.__total:
            raise Exception("not ready!!!")
        else:
            return self.__value/self.__total


class Averages:

    def __init__(self, shape, total):
        self.shape = shape
        self.__total = total
        self.__value = 0
        self.__count = 0

    def reset(self):
        self.__value = 0
        self.__count = 0

    def add(self, v):
        self.__value += v
        self.__count += 1

    def get_value(self):
        if self.__count != self.__total:
            raise Exception("not ready!!!")
        else:
            return self.__value/self.__total
