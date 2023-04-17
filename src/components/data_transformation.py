import os
import sys
sys.path.insert(0,'/Users/owner/Downloads/mlproject/src')
sys.path.insert(0,'/Users/owner/Downloads/mlproject/src/components')
from data_ingestion import *
from exception import CustomException
from logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Basic Import
import numpy as np
# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings

@dataclass
class DataTransformationConfig:
    data_ingestion = DataIngestion()
    data_ingestion_returns : tuple= data_ingestion.initiate_data_ingestion()
    data_ingestion_returns : tuple=
    models : dict= {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "XGBRegressor": XGBRegressor(), 
    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
    "AdaBoost Regressor": AdaBoostRegressor()
}

class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformationConfig()
        
    def initiate_data_transformation(self):
        logging.info("Entered the data transformation method or component")
        try:
           pass
        except Exception as e:
            raise CustomException(e,sys)
        
        
if __name__=="__main__":
    obj= DataTransformation()
    obj.initiate_data_transformation()