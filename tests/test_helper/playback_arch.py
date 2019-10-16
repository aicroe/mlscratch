from mlscratch.arch import Arch


class _PlaybackArch(Arch):

    def __init__(
            self,
            costs,
            predictions,
            validation_costs,
            validation_predictions):
        self.costs = costs
        self.predictions = predictions
        self.validation_costs = validation_costs
        self.validation_predictions = validation_predictions
        self.update_params_call_count = 0
        self.check_cost_call_count = 0

    def train_initialize(self):
        pass

    def train_finalize(self):
        pass

    def evaluate(self, dataset):
        return None

    def update_params(self, dataset, labels):
        cost = self.costs[self.update_params_call_count]
        prediction = self.predictions[self.update_params_call_count]
        self.update_params_call_count += 1
        return cost, prediction

    def check_cost(
            self,
            dataset,
            labels):
        index = self.check_cost_call_count
        validation_cost = self.validation_costs[index]
        validation_prediction = self.validation_predictions[index]
        self.check_cost_call_count += 1
        return validation_cost, validation_prediction
