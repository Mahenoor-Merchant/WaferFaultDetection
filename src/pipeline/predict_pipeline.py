import shutil
import os,sys
import pandas as pd
import pickle
from src.logger import logging

from src.exception import CustomException
import sys
from flask import request
from src.constant import *
from src.utils import MainUtils

from dataclasses import dataclass
        
        
@dataclass
class PredictionPipelineConfig:
    prediction_output_dirname: str = "predictions"
    prediction_file_name:str =  "predicted_file.csv"
    model_file_path: str = os.path.join("artifacts", "model.pkl" )
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")
    prediction_file_path:str = os.path.join(prediction_output_dirname,prediction_file_name)



class PredictionPipeline:
    def __init__(self, request: request):

        self.request = request
        self.utils = MainUtils()
        self.prediction_pipeline_config = PredictionPipelineConfig()



    def save_input_files(self)-> str:

        try:
            pred_file_input_dir = "prediction_artifacts"
            os.makedirs(pred_file_input_dir, exist_ok=True)

            input_csv_file = self.request.files["file"]
            pred_file_path = os.path.join(pred_file_input_dir, input_csv_file.filename)
            
            
            input_csv_file.save(r"D:\MLProjects\waferfaultdetection\prediction_artifacts\input_csv_file.csv")


            return pred_file_path
        except Exception as e:
            raise CustomException(e,sys)

    def predict(self, features):
        try:

            # model = self.utils.load_object(r'D:\MLProjects\waferfaultdetection\artifacts\model.pkl')
            model = MainUtils.load_object(file_path=PredictionPipelineConfig.model_file_path)
            preprocessor = MainUtils.load_object(file_path=PredictionPipelineConfig.preprocessor_path)
            transformed_x = preprocessor.transform(features)

            preds = model.predict(transformed_x)

            return preds

        except Exception as e:
            raise CustomException(e, sys)
        
    def get_predicted_dataframe(self, input_dataframe_path:pd.DataFrame):
   
        try:

            prediction_column_name : str = 'Target'
            input_dataframe: pd.DataFrame = pd.read_csv(r"D:\MLProjects\waferfaultdetection\prediction_artifacts\input_csv_file.csv")
            
            input_dataframe =  input_dataframe.drop(columns="Unnamed: 0") if "Unnamed: 0" in input_dataframe.columns else input_dataframe
            input_dataframe = input_dataframe.drop('Good/Bad', axis=1)

            predictions = self.predict(input_dataframe)
            input_dataframe[prediction_column_name] = [pred for pred in predictions]
            target_column_mapping = {0:'bad', 1:'good'}

            input_dataframe[prediction_column_name] = input_dataframe[prediction_column_name].map(target_column_mapping)
            
            os.makedirs( self.prediction_pipeline_config.prediction_output_dirname, exist_ok= True)
            input_dataframe.to_csv(self.prediction_pipeline_config.prediction_file_path, index= False)
            logging.info("predictions completed. ")



        except Exception as e:
            raise CustomException(e, sys) from e
        

        
    def run_pipeline(self):
        try:
            input_csv_path = self.save_input_files()
            self.get_predicted_dataframe(input_csv_path)

            return self.prediction_pipeline_config


        except Exception as e:
            raise CustomException(e,sys)
            
        

 
        

        