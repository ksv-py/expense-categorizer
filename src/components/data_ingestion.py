import os
import sys
from pathlib import Path
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

sys.path.append(str(Path(__file__).parent.parent))
from logger import logging
from exception import CustomException

@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion paths.
    """
    raw_data_path = os.path.join('artifacts', 'raw.csv')
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    synthetic_data_path = os.path.join('artifacts', 'synthetic-data.csv')

class DataIngestion:
    """
    Class for performing data ingestion which includes:
    - Downloading dataset from Kaggle
    - Merging with synthetic data if present
    - Basic validation of data
    - Saving raw, train, and test splits
    """
    def __init__(self):
        logging.info('Initializing DataIngestion instance.')
        self.ingestion_config = DataIngestionConfig()

    def validate_data(self, df: pd.DataFrame):
        """
        Validates the data for common issues like nulls, invalid datatypes, or duplicates.
        """
        logging.info("Validating raw dataset for nulls, types, and duplicates.")
        if df.isnull().sum().any():
            logging.warning(f"Missing values found in:")
            logging.warning(f"{df.isnull().sum()[df.isnull().sum() > 0]}")
        if df.duplicated().sum() > 0:
            logging.warning(f"Duplicate records found: {df.duplicated().sum()} - Removing duplicates.")
            df = df.drop_duplicates()
        return df

    def initiate_data_ingestion(self):
        logging.info('Starting data ingestion process.')
        try:
            logging.info('Fetching dataset from Kaggle.')
            path = kagglehub.dataset_download("prasad22/daily-transactions-dataset")
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

            if not csv_files:
                logging.error('No CSV files found in downloaded dataset.')
                raise FileNotFoundError(f'No CSV file found in {path}')

            kaggle_df = pd.read_csv(os.path.join(path, csv_files[0]))
            logging.info(f"Loaded Kaggle dataset with shape: {kaggle_df.shape}")

            if os.path.exists(self.ingestion_config.synthetic_data_path):
                synthetic_df = pd.read_csv(self.ingestion_config.synthetic_data_path)
                logging.info(f"Loaded synthetic dataset with shape: {synthetic_df.shape}")
                df = pd.concat([kaggle_df, synthetic_df], ignore_index=True)
            else:
                logging.warning("Synthetic data not found. Proceeding with only Kaggle data.")
                df = kaggle_df

            df = self.validate_data(df)
            logging.info(f"Combined dataset shape after validation: {df.shape}")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Saved raw dataset at: {self.ingestion_config.raw_data_path}")

            logging.info('Splitting data into training and testing sets.')
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info(f"Saved train data at: {self.ingestion_config.train_data_path}")
            logging.info(f"Saved test data at: {self.ingestion_config.test_data_path}")

            logging.info('Data ingestion process completed successfully.')
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error('Exception occurred during data ingestion.')
            raise CustomException(e, sys)
