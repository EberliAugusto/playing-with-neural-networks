
class Reporter():

    def __init__(self,
                 cost_min=True,
                 cost_max=True,
                 cost_avg=True,
                 accuracy=True):
        self.cost_min = cost_min
        self.cost_max = cost_max
        self.cost_avg = cost_avg
        self.accuracy = accuracy

    def report_train(self, label, trainer):
        print("-------------------TRAINING-----------------------")
        print("->", label)
        if self.cost_avg:
            print("Avg:", trainer.averages.get_value())
        if self.accuracy:
            print("Accuracy:", trainer.accuracy.get_value() * 100, "%")
        print("----------------------------------------------")
        return trainer

    def report_test(self, test,  label= "Testing"):
        print("-------------------TESTING-----------------------")
        print("->",label)
        if self.cost_min:
            print("Min:", test.min_cost())
        if self.cost_max:
            print("Max:", test.max_cost())
        if self.cost_avg:
            print("Avg:", test.average_cost())
        if self.accuracy:
            print("Accuracy:", test.accuracy() * 100, "%")
        print("----------------------------------------------")
        return test
