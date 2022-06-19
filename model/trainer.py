import joblib
import os

import numpy as np
import pandas as pd

from utils.data import load_config, load_csv
from .feature_engineering import feature_engineering
from .builder import build_regressor

from sklearn.model_selection import train_test_split


class Trainer:
    def __init__(
            self,
            config_pth
    ):
        # load configurations from config.yaml
        self.config = load_config(config_pth)
        self.eval_metric = self.config["train_config"]["eval_metric"]
        self.model_pth = os.path.join(
            self.config["main"]["save_pth"],
            self.config["main"]["regressor_filename"]
        )

        # get the train, test data as pandas dataframe
        trainset, target = load_csv(
            self.config["main"]["training_data"]
        )
        testset, _ = load_csv(
            self.config["main"]["test_data"]
        )

        # apply feature engineering
        trainset = self._expand_features(trainset)
        self.testset = self._expand_features(testset)

        # train, val split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(trainset, target)

        self.model = build_regressor(self.config["regressor"])

    def _expand_features(self, dataframe):
        dataframe = feature_engineering(self.config, dataframe)

        return dataframe

    @staticmethod
    def remove_nan(dataframe):
        col_names = dataframe.columns

        for col in col_names:
            # change NaN values to 0
            dataframe[col] = dataframe[col].fillna(0)

        return dataframe

    @staticmethod
    def nmae(true , pred):
        return np.mean((np.abs(true-pred))/true)

    def fit(self):
        self.model.fit(
            self.X_train,
            self.y_train,
            eval_metric=self.eval_metric,
            eval_set=[(self.X_test, self.y_test)],
        )

        joblib.dump(self.model, self.model_pth)

    def test(self):
        # load model weights
        self.model = joblib.load(self.model_pth)

        # get model predictions
        preds = self.model.predict(self.testset)
        preds = np.around(preds)

        # write submission file
        submission = pd.read_csv(os.path.join(
            self.config["main"]["data_pth"], "sample_submission.csv"
        ))
        submission["rental"] = preds

        submission.to_csv("submission.csv", mode='w')
