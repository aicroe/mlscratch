import numpy as np

from mlscratch import Model
from mlscratch.arch import Arch
from mlscratch.trainer import SimpleTrainer
from mlscratch.measurer import Measurer, PairwiseMeasurer

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class ThresholdMeasurer(Measurer[float]):

    def __init__(self, threshold):
        self._threshold = threshold
        self._pairwise_measurer = PairwiseMeasurer()

    def measure(self, result, expected):
        mapped_result = (result > self._threshold).astype(int)
        return self._pairwise_measurer.measure(mapped_result, expected)

class SigmoidPerceptron(Arch):

    def __init__(self, units, learning_rate=0.5):
        self._weights = np.zeros(units)
        self._bias = 0
        self._learning_rate = learning_rate

    def update_params(self, dataset, labels):
        evaluations = self.evaluate(dataset)
        error = labels - evaluations
        delta = error * (evaluations * (1 - evaluations))
        self._weights += np.sum(self._learning_rate * delta * np.transpose(dataset), axis=1)
        self._bias += np.sum(self._learning_rate * delta)
        return (np.average(error), evaluations)

    def evaluate(self, dataset):
        return sigmoid(np.sum(dataset * self._weights, axis=1) + self._bias)


model = Model(SigmoidPerceptron(2))

epochs, costs, accuracies, *_ = model.train(
    np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
    np.array([0, 0, 0, 1]),
    None,
    None,
    SimpleTrainer(),
    ThresholdMeasurer(0.5),
    None,
    epochs=35,
)

print('Epochs:     ', epochs)
print('Costs:      ', costs)
print('Accuracies: ', accuracies)
print('Predictions:', model.evaluate(np.array([[0, 0], [0, 1], [1, 0], [1, 1]])))
