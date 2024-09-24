# DI pipeline to read Dset, split in train-test and save these along with raw data
# .. while logging each step of the process
import os 
import sys
import pandas as pd

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig: # dataclass that defines file paths for storing ingested data, stored in the 'artifacts' folder
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion: # DI class handles DI process, constructor initializes of DIConfig 
    def __init__(self): 
        self.ingestion_config = DataIngestionConfig() 

    def initiate_data_ingestion(self): # method performs the actual DI process
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(r'notebook\data\StudentsPerformance.csv') # read .csv to dataframe 
            logging.info("Read the datset as Dataframe") 
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) # create dirs for storing data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) # raw data saved as CSV in the path specified
            
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True) # saving train data as csv in paths specfied
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True) # saving test data..

            logging.info("Ingestion of the data has been completed")

            return(
                self.ingestion_config.train_data_path, # return train & test file paths
                self.ingestion_config.test_data_path,
            )
        except Exception as e: 
            raise CustomException(e, sys) # if exception occurs, it's caught and raised as CE, includes addn'l errorhandling and logging
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr, test_arr)

    print(model_trainer.initiate_model_trainer(train_arr, test_arr))

