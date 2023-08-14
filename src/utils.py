import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


'''
Function to be used to save the model and data pre-processing objects
input:
    - file_path: path of the file to save the object to
    - obj: object to be saved
postcondition:
    - the object is saved in the specified file path
'''

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
    
    
'''
Function to evaluate the best model to be used in training
input:
    - X_train: training dataset
    - y_train: target feature dataset to be predicted during training
    - X_test: testing dataset
    - y_test: target feature dataset to be predicted during testing
    - models: type: dict ; dictionary of models to be assessed during training and testing 
    - params: models' hyperparameters dictionary
output:
    - report: type: dict ; dictionary containing a report of the models' performance during training and testing
'''
def evaluate_models(X_train, y_train,X_test,y_test,models,params):
    try:
        report = {}

        for i in range(len(list(models))):
            # select model and hyperparameters: 
            model = list(models.values())[i]
            param = params[list(models.keys())[i]] 

            gs = GridSearchCV(model,param,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)

            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
    
'''
Function to load pickeled object from a given file path
input:
    - file_path: path to the file from which we load the desired object

'''
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)