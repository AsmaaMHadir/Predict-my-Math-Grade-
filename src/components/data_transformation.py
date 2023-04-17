import os
import sys
sys.path.insert(0,'/Users/owner/Downloads/mlproject/src')

from data_ingestion import *
from exception import CustomException
from logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass,  field
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
from typing import Dict


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square

@dataclass
class DataTransformationConfig:
    
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")
    
    models: Dict[str, int] = field(default_factory=lambda: {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "XGBRegressor": XGBRegressor(), 
    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
    "AdaBoost Regressor": AdaBoostRegressor()
})
    

class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformationConfig()
        df = pd.read_csv(self.transformation_config.train_data_path)
        logging.info("completed reading the dataset!")
        
        self.X = df.drop(columns=['math score'],axis=1)
        self.y = df['math score']
        
        
        
    

        
    def initiate_data_transformation(self):
        try:
            logging.info("Entered the data transformation method or component")
            df = pd.read_csv(self.transformation_config.train_data_path)
            logging.info("completed reading the dataset!")
            
            
            
            num_features = self.X.select_dtypes(exclude="object").columns
            cat_features = self.X.select_dtypes(include="object").columns
            
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            
            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, cat_features),
                    ("StandardScaler", numeric_transformer, num_features),        
                ]
            )
            logging.info("completed applying the transformers")
            self.X = preprocessor.fit_transform(self.X)
        
            X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
            logging.info("training and testing datasets are ready!")
            
            model_list = []
            r2_list =[]

            for i in range(len(list(self.transformation_config.models))):
                model = list(self.transformation_config.models.values())[i]
                model.fit(X_train, y_train) # Train model

                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Evaluate Train and Test dataset
                model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)

                model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

                
                print(list(self.transformation_config.models.keys())[i])
                model_list.append(list(self.transformation_config.models.keys())[i])
                
                print('Model performance for Training set')
                print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
                print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
                print("- R2 Score: {:.4f}".format(model_train_r2))

                print('----------------------------------')
                
                print('Model performance for Test set')
                print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
                print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
                print("- R2 Score: {:.4f}".format(model_test_r2))
                r2_list.append(model_test_r2)
                
                print('='*35)
                print('\n')
        except Exception as e:
            raise CustomException(e,sys)
        
        
if __name__=="__main__":
    obj= DataTransformation()
    obj.initiate_data_transformation()