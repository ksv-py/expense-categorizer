import os
from pathlib import Path
import sys
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
sys.path.append(str(Path(__file__).parent.parent))

from logger import logging
from exception import CustomException

@dataclass
class DataIngestionConfig:
    
    raw_data_path = os.path.join('artifacts','raw.csv')
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    synthetic_data_path = os.path.join('artifacts','synthetic-data.csv')

class DataIngestion:

    def __init__(self):
        logging.info('Initializing Instance for Data Ingestion Config.')
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Process started.')

        try:
            logging.info('Fetching Raw Data from Kaggle.')
            path = kagglehub.dataset_download("prasad22/daily-transactions-dataset")
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

            if not csv_files:
                logging.error(f'No CSV File found.')
                raise FileNotFoundError(f'No CSV File found at {path}')
            
            synthetic_df = pd.read_csv(self.ingestion_config.synthetic_data_path)

            logging.info(f'Kaggle Dataset stored at - {path}')
            df = pd.read_csv(os.path.join(path,csv_files[0]))
            df = pd.concat([df, synthetic_df ], ignore_index=True) 

            logging.info('Creating Raw Data Path.')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f'Stored Raw data at - {self.ingestion_config.raw_data_path}')

            logging.info('Creating Train/Test Data Path.')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            
            logging.info('Splitting Data Sets into Train and Test Set.')
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

            train_data.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            logging.info(f'Stored Train data at - {self.ingestion_config.train_data_path}')
            test_data.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info(f'Stored Train data at - {self.ingestion_config.test_data_path}')

            logging.info('Data Ingestion Process Successful.')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            logging.error("ERROR in executing the script")
            raise CustomException(e,sys)