import numpy as np
from nn.utils import EWMA


class GradientCalculator:

    def __init__(self, cost):
        self.cost = cost

    def calculate(self, nn, z_by_layer, a_by_layer, input_, desired_output):
        cost_prime_on_w = [None] * nn.depth

        activation_prime = [nn.activation[l].prime(z) for l, z in enumerate(z_by_layer)]

        cost_prime_on_a = self.__cost_prime_on_a(nn, a_by_layer, activation_prime, desired_output)

        hold_it = [activation_prime[l] * cost_prime_on_a[l] for l in range(nn.depth)]

        for l in range(1, nn.depth):
            cost_prime_on_w[l] = np.outer(hold_it[l], a_by_layer[l - 1])

        cost_prime_on_w[0] = np.outer(hold_it[0], input_)

        cost_prime_on_b = hold_it  # (hold_it * 1) just for clarity
        return cost_prime_on_w, cost_prime_on_b

    def __cost_prime_on_a(self, nn, a_by_layer, sig_prime, y):
        cost_prime_on_a = [None] * nn.depth
        cost_prime_on_a[-1] = self.cost.prime(a_by_layer[-1], y)

        # Iterating from the last but one layer till the first layer.
        for l in range(nn.depth - 2, -1, -1):
            hold_it = sig_prime[l + 1] * cost_prime_on_a[l + 1]
            cost_prime_on_a[l] = hold_it @ nn.w_by_layer[l + 1]

        return cost_prime_on_a


class GradientDescent:

    def __init__(self, lr, cost):
        self.lr = lr
        self.cost = cost
        self.grad_calc = GradientCalculator(cost)

    def initialize(self, nn):
        return self

    def calculate_gradients(self, nn, z_by_layer, a_by_layer, input_, desired_output):
        return self.grad_calc.calculate(nn, z_by_layer, a_by_layer, input_, desired_output)

    def update_network(self, nn, cost_prime_on_w, cost_prime_on_b):
        for l in range(nn.depth):
            lr = self.lr.get_value()
            nn.w_by_layer[l] = nn.w_by_layer[l] - lr * cost_prime_on_w[l]
            nn.b_by_layer[l] = nn.b_by_layer[l] - lr * cost_prime_on_b[l]


class GradientDescentWithMomentum:

    def __init__(self, lr, cost, friction=0.9, correct=True):
        self.lr = lr
        self.friction = friction
        self.correct = correct
        self.cost = cost
        self.grad_calc = GradientCalculator(cost)

    def initialize(self, nn):
        self.momemtum_w = [EWMA((outputSize, inputSize), self.friction, self.correct) for outputSize, inputSize in
                           nn.layers]
        self.momemtum_b = [EWMA(outputSize, self.friction, self.correct) for outputSize, _ in nn.layers]
        return self

    def calculate_gradients(self, nn, z_by_layer, a_by_layer, input_, desired_output):
        return self.grad_calc.calculate(nn, z_by_layer, a_by_layer, input_, desired_output)

    def update_network(self, nn, cost_prime_on_w, cost_prime_on_b):
        for l in range(nn.depth):
            self.momemtum_w[l].add_values(cost_prime_on_w[l])
            self.momemtum_b[l].add_values(cost_prime_on_b[l])

            lr = self.lr.get_value()
            nn.w_by_layer[l] = nn.w_by_layer[l] - lr * self.momemtum_w[l].get()
            nn.b_by_layer[l] = nn.b_by_layer[l] - lr * self.momemtum_b[l].get()


# Root Mean Square Prop
class RMSProp:

    def __init__(self, lr, cost, b=0.9999, tiny_number=1.e-8, correct=True):
        self.lr = lr
        self.b = float(b)
        self.correct = correct
        self.cost = cost
        self.tiny_number = float(tiny_number)
        self.grad_calc = GradientCalculator(cost)

    def initialize(self, nn):
        self.square_avg_w = [EWMA((o, i), self.b, self.correct) for o, i in nn.layers]
        self.square_avg_b = [EWMA(o, self.b, self.correct) for o, _ in nn.layers]
        return self

    def calculate_gradients(self, nn, z_by_layer, a_by_layer, input_, desired_output):
        return self.grad_calc.calculate(nn, z_by_layer, a_by_layer, input_, desired_output)

    def update_network(self, nn, cost_prime_on_w, cost_prime_on_b):
        for l in range(nn.depth):
            self.square_avg_w[l].add_values(cost_prime_on_w[l] ** 2)
            self.square_avg_b[l].add_values(cost_prime_on_b[l] ** 2)

            lr = self.lr.get_value()

            nn.w_by_layer[l] = nn.w_by_layer[l] - lr * cost_prime_on_w[l] / (
                    np.sqrt(self.square_avg_w[l].get()) + self.tiny_number)
            nn.b_by_layer[l] = nn.b_by_layer[l] - lr * cost_prime_on_b[l] / (
                    np.sqrt(self.square_avg_b[l].get()) + self.tiny_number)


# Adaptive moment estimation
class Adam:

    def __init__(self, lr, cost, m_b=0.9, s_b=0.9999, tiny_number=1.e-8):
        self.lr = lr
        self.m_b = float(m_b)
        self.s_b = float(s_b)
        self.cost = cost
        self.tiny_number = float(tiny_number)
        self.grad_calc = GradientCalculator(cost)

    def initialize(self, nn):
        self.momemtum_w = [EWMA((o, i), self.m_b, True) for o, i in nn.layers]
        self.momemtum_b = [EWMA(o, self.m_b, True) for o, _ in nn.layers]
        self.square_avg_w = [EWMA((o, i), self.s_b, True) for o, i in nn.layers]
        self.square_avg_b = [EWMA(o, self.s_b, True) for o, _ in nn.layers]
        return self

    def calculate_gradients(self, nn, z_by_layer, a_by_layer, input_, desired_output):
        return self.grad_calc.calculate(nn, z_by_layer, a_by_layer, input_, desired_output)

    def update_network(self, nn, cost_prime_on_w, cost_prime_on_b):
        for l in range(nn.depth):
            self.momemtum_w[l].add_values(cost_prime_on_w[l])
            self.momemtum_b[l].add_values(cost_prime_on_b[l])
            self.square_avg_w[l].add_values(cost_prime_on_w[l] ** 2)
            self.square_avg_b[l].add_values(cost_prime_on_b[l] ** 2)

            lr = self.lr.get_value()
            nn.w_by_layer[l] = nn.w_by_layer[l] - lr * self.momemtum_w[l].get() / (
                    np.sqrt(self.square_avg_w[l].get()) + self.tiny_number)
            nn.b_by_layer[l] = nn.b_by_layer[l] - lr * self.momemtum_b[l].get() / (
                    np.sqrt(self.square_avg_b[l].get()) + self.tiny_number)



