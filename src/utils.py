import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle 
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj): # create dir, open file, save in specific file path
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    """
    Load a Python object from a file using dill serialization.

    Args:
        file_path (str): The path to the file containing the serialized object.

    Returns:
        object: The deserialized Python object.

    Raises:
        CustomException: If an error occurs during file reading or deserialization.

    This function is used to load saved models, preprocessors,
    or other Python objects that have been serialized using dill.
    """
    try:
        # Open file in binary read mode 
        with open(file_path, "rb") as file_obj:
            # Use dill to desearlize and load the object from file
            return pickle.load(file_obj)
    except Exception as e:
        # If any exception occors duing file opening/desearlization raise CE
        raise CustomException(e, sys)