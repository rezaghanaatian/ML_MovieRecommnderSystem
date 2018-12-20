import time

from helpers import load_dataset, create_submission_file
from models_blender import Blender
from models_knn import SurpriseKNN
from models_als import ALSOptimizer
from models_svd import SurpriseSVD
from models_funcpyfm import factorization_machine_pyfm

start = time.time()
print("In the name of God\n============\n[LOG] START\n============")

# Load Datasets
path_dataset = "data/data_train.csv"
path_test_dataset = "data/sample_submission.csv"

train_df = load_dataset(path_dataset).head(20)
test_df = load_dataset(path_test_dataset).head(10)

prediction_models = []
best_weights = []

knn = SurpriseKNN(k=90, user_based=False)
knn_weight = 0.3
prediction_models.append(knn)
best_weights.append(knn_weight)

svd = SurpriseSVD()
prediction_models.append(svd)
svd_weight = 0.3
best_weights.append(svd_weight)

als = ALSOptimizer()
prediction_models.append(als)
als_weight = 0.1
best_weights.append(als_weight)

# This line can be used for finding the best weight for each algorithm, but running it takes a long time.
# best_weights = Blender.tune_weights(prediction_models, train_df)

blender_model = Blender(models=prediction_models, weights=best_weights)
blender_model.fit(train_df)
predictions = blender_model.predict(test_df)

# Since PyFm implementation is different from other models, its prediction will be added separately
pred_pyfm = factorization_machine_pyfm(train_df, test_df)
pyfm_weight = 0.3
predictions.Prediction += pyfm_weight * pred_pyfm.Prediction

print("============\n[LOG] SAVE RESULT IN CSV FILE\n============")
create_submission_file(predictions, "output.csv", round_predictions=False)
