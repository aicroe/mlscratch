from mlbatch.train_watcher import TrainWatcher


class _TrainWatcherRecorder(TrainWatcher):

    def __init__(self):
        self.epochs = []
        self.accuracies = []
        self.costs = []
        self.validation_epochs = []
        self.validation_accuracies = []
        self.validation_costs = []

    def on_epoch(self, epoch, cost, accuracy):
        self.epochs.append(epoch)
        self.costs.append(cost)
        self.accuracies.append(accuracy)

    def on_validation_epoch(self, epoch, cost, accuracy):
        self.validation_epochs.append(epoch)
        self.validation_costs.append(cost)
        self.validation_accuracies.append(accuracy)
