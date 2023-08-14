import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:

    def __init__(self):
        pass

    '''
    Function to return the model's prediction
    input:
        - features: input features given by the users and fed to the model
    output:
        - preds: model's resulting predictions
    '''
    def predict(self, features):
        try:
            

            model_relative_path = "model.pkl"
            preprocessor_relative_path = "proprocessor.pkl"
                
                
            model_path = os.path.join('artifacts', model_relative_path)
            preprocessor_path = os.path.join('artifacts', preprocessor_relative_path)

            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            # transform the user's input features
            data_scaled = preprocessor.transform(features)
            
            # return the model's predictions
            preds=model.predict(data_scaled)
                
        except Exception as e:
            raise CustomException(e,sys)

        return preds


class CustomData:
    def __init__(self,
                 gender:str,
                 race_ethnicity,
                 parental_level_of_education,
                 lunch: int,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.reading_score = reading_score
        self.writing_score = writing_score
        self.test_preparation_course = test_preparation_course
        
    # return input as dataframe object for the model to process
    def get_data_As_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender":[self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch":[self.lunch],
                "reading score": [self.reading_score],
                "writing score":[self.writing_score],
                "test preparation course":[self.test_preparation_course],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
        