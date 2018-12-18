import time
import scipy.optimize as sco

from prediction_model import PredictionModel


class Blender(PredictionModel):
    """
       Blend different models for recommendation.
    """

    def __init__(self, weights, models):
        super(Blender, self).__init__()
        if len(models) != len(weights):
            raise Exception('Size of models and weights should be the same!')

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
            prediction = model.predict(test_df)
            predictions.append(prediction)
            print(prediction.head())
            print("[LOG] Prediction by {0} completed in {1}".format(model.get_name(), time.time() - tt))

        output = test_df.copy()
        output.Prediction = 0

        # Use each model's weight to generate the final prediction
        for i in range(len(self.models)):
            output.Prediction += self.weights[i] * predictions[i].Prediction
            print("[LOG] Improving predictions by {0} completed".format(self.models[i].get_name()))

        def round_prediction(row):
            value = row.Prediction
            value = 5 if value > 5 else value
            value = 1 if value < 1 else value
            return value

        output['Prediction'] = output.apply(round_prediction, axis=1)
        return output

    @staticmethod
    def tune_weights(models, train_df):
        """
        tune weights of models in blending
        :param models: models to be used in blending
        :param train_df: train data for calculating RMSE by cross-validation
        :return: the found best weights for given models
        """
        num = len(models)
        init_weights = []
        for i in range(len(models)):
            init_weights.append(1 / num)

        result = sco.minimize(evaluate_weights, init_weights, args=(models, train_df),
                              options={'maxiter': 5, 'disp': True})
        return result.x


def evaluate_weights(weights, models, train_df):
    blender = Blender(models=models, weights=weights)
    blender.fit(train_df)
    rmse = blender.cross_validate()
    # print("{0} --- > {1}".format(weights, rmse))
    return rmse
