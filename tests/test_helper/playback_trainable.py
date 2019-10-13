from mlscratch.trainable import Trainable


class _PlaybackTrainable(Trainable):

    def __init__(
            self,
            costs,
            accuracies,
            validation_costs,
            validation_accuracies):
        self.costs = costs
        self.accuracies = accuracies
        self.validation_costs = validation_costs
        self.validation_accuracies = validation_accuracies
        self.update_params_call_count = 0
        self.evaluate_validation_set_call_count = 0

    def update_params(self, dataset, labels):
        cost = self.costs[self.update_params_call_count]
        accuracy = self.accuracies[self.update_params_call_count]
        self.update_params_call_count += 1
        return cost, accuracy

    def evaluate_validation_set(
            self,
            validation_dataset,
            validation_labels):
        index = self.evaluate_validation_set_call_count
        validation_cost = self.validation_costs[index]
        validation_accuracy = self.validation_accuracies[index]
        self.evaluate_validation_set_call_count += 1
        return validation_cost, validation_accuracy
