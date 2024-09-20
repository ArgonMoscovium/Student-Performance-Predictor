# Ceate prediction pipeline using Flask Web App, 
#... interacs with .pkl files to predict student perormance
import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object # to load .plk files

class PredictPipeline:
    '''
    A class for making preds using a pre-trained model and preprocessor
    '''
    def __init__(self):
        pass 

    def predict(self, features):
        """
        Make predictions on the input features.
        
        Args:
            features (pandas.DataFrame): Input features for prediction.
        
        Returns:
            numpy.ndarray: Predicted values.
        
        Raises:
            CustomException: If an error occurs during prediction.
        """
        try:
            # Define paths to the saved model and preprocessor
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            print("Before Loading")
            # Load the saved model and preprocessor
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            
            # Make predictions using the loaded model
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
                
        except Exception as e:
            # If error exists, raise CE
            raise CustomException(e, sys)

class CustomData:
    """
    A class to encapsulate input data from Flask app and convert it to a pandas df.
    """
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):
        """
        Initialize the CustomData object with input features from Flask app
        
        Args:
            gender (str): Gender of the student.
            race_ethnicity (str): Race/ethnicity of the student.
            parental_level_of_education (str): Highest education level of the student's parents.
            lunch (str): Type of lunch the student receives.
            test_preparation_course (str): Whether the student completed a test preparation course.
            reading_score (int): Student's reading score.
            writing_score (int): Student's writing score.
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """
        Convert the input data from Flask app to a pandas df
        
        Returns:
            pandas.DataFrame: A DataFrame containing the input data to train the model
        
        Raises:
            CustomException: If an error occurs during DataFrame creation.
        """
        try:
            # Create dict with ip data
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            # Convert dict to df
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            # If error occors, raise CE
            raise CustomException(e, sys)
