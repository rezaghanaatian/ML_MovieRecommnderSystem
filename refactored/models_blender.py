import time
from refactored.prediction_model import PredictionModel


class Blender(PredictionModel):
    """
       Blend different models for recommendation.
    """

    def __init__(self, weights, models):
        super(Blender, self).__init__()
        if len(models) != len(weights):
            print("Warning!\n Size of models and weights should be the same!")

        self.weights = weights
        self.models = models

    def predict(self, test_df):

        predictions = []

        for model in self.models:

            if not issubclass(type(model), PredictionModel):
                continue

            print("[LOG] Preparing model === {0} ===".format(model.get_name()))
            tt = time.time()
            model.fit(self.train_df)
            predictions.append(model.predict(test_df))
            print("[LOG] Prediction by {0} completed in {1}".format(model.get_name(), time.time() - tt))

        output = test_df.copy()

        # Use each model's weight to generate the final prediction
        for i in range(len(self.models)):
            model = self.models[i]
            prediction = predictions[i]
            output.Prediction += self.weights[model.get_name()] * prediction.Prediction

        def round_prediction(row):
            value = round(row.Prediction)
            value = 5 if value > 5 else value
            value = 1 if value < 1 else value
            return value

        output['Prediction'] = output.apply(round_prediction, axis=1)
        return output
