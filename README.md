# MlScratch
A small Machine Learning python library. Exposes a set of classes that helps to build and reuse models for any purpose.

## Installation

Requires Python 3.6 or higher

```bash
$ pip install -e git+git@github.com:aicroe/mlscratch.git@master#egg=mlscratch
```

## Getting started

The main class is `Model`, it exposes a set of common operations like `train` or `evaluate` but it's not limited so could be easily extended. In order to use it you first need to build an `Arch` instance. The arch is the one who actually implements the  model's engine and design.

Take a look to the next code snippet, it shows an example of a perceptron. Notice that in order to write an `Arch` you need to extend it and overwrite at least two methods: The one that evaluates input data and the one who optimizes the parameters.

```python
import numpy as np
from mlscratch.arch import Arch

class Perceptron(Arch):

    def __init__(self, units, threshold=0.5, learning_rate=0.1):
        self._weights = np.zeros(units)
        self._threshold = threshold
        self._learning_rate = learning_rate

    def update_params(self, dataset, labels):
        evaluations = self.evaluate(dataset)
        error = labels - evaluations
        self._weights += np.sum(self._learning_rate * error * np.transpose(dataset), axis=1)

        # Returns the loss and evaluations
        return (np.average(error), evaluations)

    def evaluate(self, dataset):
        sum_ = np.sum(dataset * self._weights, axis=1)
        return (sum_ > self._threshold).astype(int)
```

Once you have the `Arch` instance ready, you can begin to train it. In order to to so, a couple of classes must come in to play. You'll need to instantiate a `Trainer` and a `Measurer`, optionally you could instantiate a `TrainWatcher` too but it was left out of the example for simplicity. With specializations of those classes you can fulfill the model's train contract and start the training.

`mlscratch` already ships with implementations of these abstractions, ready to use out of the box, but you can create your own ones if wanted.

```python
from mlscratch import Model
from mlscratch.trainer import SimpleTrainer
from mlscratch.measurer import PairwiseMeasurer

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
```

After the training finishes you can test the model or inspect the training traces. Notice `evaluate` as well as `train` work over data batches.

```python
print('Epochs:     ', epochs)
print('Costs:      ', costs)
print('Accuracies: ', accuracies)
print('Predictions:', model.evaluate(np.array([[0, 0], [0, 1], [1, 0], [1, 1]])))
```

You can find more at the [examples](examples) folder.

## Developing

**Setup your environment**

* Clone the project
* Create a dedicated virtual environment
```bash
$ cd mlscratch
$ python3 -m venv python3
$ source python3/bin/activate
```
* Install the dependencies
```bash
$ pip install -e ".[dev]"
```

**Install for development**
```bash
$ python setup.py develop
```

This stages the code at `sys.path`, so that way the code could be tested from the interactive console.

**Run the tests**
```bash
python setup.py test
```

**Run the linter**
```bash
pylint src
```

The above will output a friendly colorized report, if needed it can be avoided by appending the option: `--output-format=text`
