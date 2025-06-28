from dotenv import load_dotenv
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path
from pymongo import MongoClient
sys.path.append(str(Path(__file__).parent.parent))

from components.model_trainer import ModelTrainer
from components.data_transformation import DataTransformation
from components.data_ingestion import DataIngestion

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client[os.getenv("MONGO_DB_NAME")]
collection = db[os.getenv("MONGO_COLLECTION_NAME")]

@dataclass
class TrainPipelineConfig:
    original_raw_path = os.path.join('artifacts','raw.csv')
    synthetic_data_path = os.path.join('artifacts','synthetic-data.csv')
    retrain_data_path = os.path.join('artifacts','retrain.csv')
    retest_data_path = os.path.join('artifacts','retest.csv')

class TrainPipeline:
    def __init__(self):
        self.train_pipeline_config = TrainPipelineConfig()

    def initiate_train_pipeline(self):
        user_df = pd.DataFrame(list(collection.find()))
        
        if user_df.empty:
            raise FileNotFoundError("User feedback data is missing.")

        required_cols = {"Amount", "Mode", "Subcategory", "Category"}
        missing = required_cols - set(user_df.columns)
        if missing:
            raise ValueError(f"Missing required columns in user feedback: {missing}")
        
        amplified_df = pd.concat([user_df] * 20, ignore_index=True)

        if os.path.exists(self.train_pipeline_config.original_raw_path):
            original_raw_df = pd.read_csv(self.train_pipeline_config.original_raw_path)
            synthetic_df = pd.read_csv(self.train_pipeline_config.synthetic_data_path)
            original_df = pd.concat([original_raw_df, synthetic_df], ignore_index=True)
            original_df = DataIngestion().validate_data(original_raw_df)
            df = pd.concat([original_df, amplified_df], ignore_index=True)
            
        else:
            df = amplified_df.copy()
        df['Category'] = df['Category'].astype(str).str.lower().str.strip()

        # Drop categories with fewer than 2 samples
        category_counts = df['Category'].value_counts()
        valid_categories = category_counts[category_counts >= 2].index.tolist()
        df = df[df['Category'].isin(valid_categories)].copy()

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_df, test_df = train_test_split(df, random_state=42,  test_size=0.2, stratify=df['Category'])

        train_df.to_csv(self.train_pipeline_config.retrain_data_path, index=False)
        test_df.to_csv(self.train_pipeline_config.retest_data_path, index=False)

if __name__ == "__main__":
    TrainPipeline().initiate_train_pipeline()
    
    transformer = DataTransformation()
    train_arr, test_arr, sample_weights = transformer.initiate_data_transformation(
        TrainPipelineConfig().retrain_data_path,
        TrainPipelineConfig().retest_data_path)
    
    accuracy = ModelTrainer().initiate_model_training(train_arr, test_arr, sample_weights=sample_weights)
    print(f"Final selected model accuracy: {accuracy:.2f}")
