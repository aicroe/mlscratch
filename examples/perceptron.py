import numpy as np

from mlscratch import Model
from mlscratch.arch import Arch
from mlscratch.trainer import SimpleTrainer
from mlscratch.measurer import PairwiseMeasurer


class Perceptron(Arch):

    def __init__(self, units, threshold=0.5, learning_rate=0.1):
        self._weights = np.zeros(units)
        self._threshold = threshold
        self._learning_rate = learning_rate

    def update_params(self, dataset, labels):
        evaluations = self.evaluate(dataset)
        error = labels - evaluations
        self._weights += np.sum(self._learning_rate * error * np.transpose(dataset), axis=1)
        return (np.average(error), evaluations)

    def evaluate(self, dataset):
        sum_ = np.sum(dataset * self._weights, axis=1)
        return (sum_ > self._threshold).astype(int)


model = Model(Perceptron(2))

epochs, costs, accuracies, *_ = model.train(
    np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
    np.array([0, 0, 0, 1]),
    None,
    None,
    SimpleTrainer(),
    PairwiseMeasurer(),
    None,
    epochs=10,
)

print('Epochs:     ', epochs)
print('Costs:      ', costs)
print('Accuracies: ', accuracies)
print('Predictions:', model.evaluate(np.array([[0, 0], [0, 1], [1, 0], [1, 1]])))
