
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from exceptions import customexception  # Update with the correct import path
from logger import logging  # Update with the correct import path

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifact', 'train.csv')
    test_data_path: str = os.path.join('artifact', 'test.csv')
    raw_data_path: str = os.path.join('artifact', 'data.csv')

class DataIngestion:

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data processing initiated")
        try:
            df = pd.read_csv("notebook\data\stud.csv")  # Update with the correct data path
            logging.info("Read the DataFrame")

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)

            logging.info("train_test_split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=40)

            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is completed")
            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            raise customexception(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    data_paths = obj.initiate_data_ingestion()
    print(data_paths)  # Print the paths of the ingested data files
