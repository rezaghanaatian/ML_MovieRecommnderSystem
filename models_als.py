import os
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS
from pyspark.sql import SQLContext
from prediction_model import PredictionModel


class ALSOptimizer(PredictionModel):
    """
        A class for using surprise library
    """
    rank = 8
    lambda_ = 0.07
    iterations = 24
    
    def __init__(self, rank = rank, lambda_ = lambda_, iterations = iterations):
        super(ALSOptimizer, self).__init__()
        self.model = None
        self.rank = rank
        self.lambda_ = lambda_
        self.iterations = iterations


    def predict(self, test):
        """
        Define the global als model for recommendation.

        Args:
            train (Pandas Dataframe) : train dataset
            test (Pandas Dataframe): test dataset

        Returns:
            output (Pandas Dataframe): test dataset with updated predictions calculated with global mean
        """
        if self.model:
            self.model.fit(self.train_df.build_full_trainset())
        else:
            print("model has not been initialized.")
            
        print("[LOG] Starting Spark...")

        # configure and start spark
        conf = (SparkConf()
                .setMaster("local")
                .setAppName("My app")
                .set("spark.executor.memory", "1g")
                )
        sc = SparkContext(conf=conf)

        # test if spark works
        if sc is not None:
            print("[LOG] Spark successfully initiated!")
        else:
            print("[ERROR] Problem with spark, check your configuration")
            exit()

        # Hide spark log information
        sc.setLogLevel("ERROR")

        output = test.copy()
        
        self.train_df['Rating'] = self.train_df['Prediction']
        output ['Rating'] = output['Prediction']
        
        print("[LOG] Starting Spark...")

        # Delete folders that cause trouble while running the code
        os.system('rm -rf metastore_db')
        os.system('rm -rf __pycache__')
        
       
        self.train_df.Movie = self.train_df.Movie.astype(int)
        self.train_df.Prediction = self.train_df.Prediction.astype(int)
        self.train_df.Rating = self.train_df.Rating.astype(int)
        
        # Prepare the dataFrame to be used in ALS object instantiation with headings
        # ['index','Prediction',User','Movie','Rating']
        self.train_df = self.train_df.drop(['Prediction'], axis=1)
        output = output.drop(['Prediction'], axis=1)
        
        # Convert pd.DataFrame to Spark.rdd 
        sqlContext = SQLContext(sc)

        train_sql = sqlContext.createDataFrame(self.train_df).rdd
        test_sql = sqlContext.createDataFrame(output).rdd

        # Train the model
        print("[LOG] ALS training started; this may take a while!")
        model = ALS.train(train_sql, rank=self.rank, lambda_=self.lambda_, iterations=self.iterations)

        # Predict
        to_predict = test_sql.map(lambda p: (p[0], p[1]))
        predictions = model.predictAll(to_predict).map(lambda r: ((r[0], r[1]), r[2]))

        # Convert Spark.rdd back to pd.DataFrame
        output = predictions.toDF().toPandas()
        

        # Postprocesse  database
        output['User'] = output['_1'].apply(lambda x: x['_1'])
        output['Movie'] = output['_1'].apply(lambda x: x['_2'])
        output['Rating'] = output['_2']
        output = output.drop(['_1', '_2'], axis=1)
        output['Prediction'] = output['Rating']
        output = output.sort_values(by=['Movie', 'User'])
        output.index = range(len(output))
      
        
        def round_pred(row):
            return round(row.Prediction)
        
        output['Prediction'] = output.apply(round_pred, axis=1)
        output['Rating'] = output['Prediction']
       
        return output