import os
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

# @dataclass decorator automatically generates several special methods for the class, 
# ...such as __init__(), __repr__(), and __eq__(). This saves you from writing boilerplate code (repetitive code)
# Additionally, if you needed to add more configuration parameters in the future, 
# ...using @dataclass would make it very easy to do so without having to modify an __init__ method.
@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl") # define path for saving preprocessor object, Without @dataclass, you would need to write an __init__ method explicitly:

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for creating .pkl files 
        for data transformation
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"] # define num & cat columns
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            # create pipeline for num features
            num_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="median")), # handle missing values by imputing with median, coz: outliers
                ("scaler", StandardScaler()) # standardize features by subtra mean & div by variance
                ]
            )            
            
            # create pipeline for cat features
            cat_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="most_frequent", fill_value="missing")), # impute with most freq value
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")), # encode as one-hot numeric array
                ("scaler", StandardScaler(with_mean=False)) # False: coz data is sparse after one-hot coding
                ]
            )            
            logging.info(f"Categorical columns: {categorical_columns}") # Log column information
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine num & cat pipelines
            preprocessor = ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path): # Inputs from data_ingestion.py
        try:
            train_df = pd.read_csv(train_path) # Read train & test data
            test_df = pd.read_csv(test_path)
            logging.info("Reading of train & test data has been completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # separate features and targets
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine preprocessed features with target variable
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")
            # Saving the preprocessing object .pkl  
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        
