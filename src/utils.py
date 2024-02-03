import os
import sys

import boto3
import dill
import numpy as np
import pandas as pd
import yaml

from pymongo import MongoClient

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from src.exception import CustomException


def export_collection_as_dataframe(collection_name, db_name):
    try:
        uri='mongodb+srv://merchant0710:Noori0710@sensordata.sfntoji.mongodb.net/'

        client = MongoClient(uri)
        
        client.admin.command('ping')

        db=client['DATABASE']
        
        collection_sensor=db['WAFERFAULTDETECTION']

        df = pd.DataFrame(list(collection_sensor.find()))

        df = df.drop(columns=["_id",'Unnamed: 0'], axis=1)

        df.replace({"na": np.nan}, inplace=True)

        return df

    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)





def download_model(bucket_name, bucket_file_name, dest_file_name):
    try:
        s3_client = boto3.client("s3")

        s3_client.download_file(bucket_name, bucket_file_name, dest_file_name)

        return dest_file_name

    except Exception as e:
        raise CustomException(e, sys)





def evaluate_models(X, y, models):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


class MainUtils:
    def __init__(self) -> None:
        pass

    def read_yaml_file(self, filename: str) -> dict:
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise CustomException(e, sys) from e

    def read_schema_config_file(self) -> dict:
        try:
            schema_config = self.read_yaml_file(os.path.join("config", "schema.yaml"))

            return schema_config

        except Exception as e:
            raise CustomException(e, sys) from e
        
    def load_object(file_path):
        try:
            with open(file_path, "rb") as file_obj:
                return dill.load(file_obj)

        except Exception as e:
            raise CustomException(e, sys)


