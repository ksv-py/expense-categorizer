import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import random
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.class_weight import compute_sample_weight

from exception import CustomException
from logger import logging
from utils import save_object


# --------------------- Custom Transformers ---------------------

class GeneralCleaner(BaseEstimator, TransformerMixin):
    """Cleans 'Mode' column using regex replacements"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X['Mode'] = X['Mode'].replace(
            to_replace=r'Saving Bank account [0-9]',
            value='Netbanking',
            regex=True
        ).replace(
            to_replace=r'Equity Mutual Fund [A-Z]',
            value='Investment',
            regex=True
        ).replace(
            to_replace=r'Share Market Trading|Fixed Deposit|Recurring Deposit',
            value='Investment',
            regex=True
        )
        return X


class SubcategoryImputer(BaseEstimator, TransformerMixin):
    """Fills missing subcategories randomly from the same category"""
    def fit(self, X, y=None):
        grouped = X.groupby(['Category', 'Subcategory']).size().reset_index(name='count')
        self.sub_cat = grouped.groupby('Category')['Subcategory'].apply(list).to_dict()
        return self

    def transform(self, X):
        X = X.copy()

        def impute(row):
            if pd.isna(row['Subcategory']):
                options = self.sub_cat.get(row['Category'], [])
                return random.choice(options) if options else row['Category']
            return row['Subcategory']

        X['Subcategory'] = X.apply(impute, axis=1)
        return X


# --------------------- Config ---------------------

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join('artifacts', 'preprocessor.pkl')
    label_encoder_path = os.path.join('artifacts', 'label_encoder.pkl')


# --------------------- Transformation Class ---------------------

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformer_object(self):
        try:
            numerical_features = ['Amount']
            categorical_features = ['Mode', 'Subcategory']

            # Numerical pipeline
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            # Categorical pipeline
            cat_pipeline = Pipeline([
                ('general_cleaner', GeneralCleaner()),
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ])

            # Combine pipelines
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_features),
                ('cat_pipeline', cat_pipeline, categorical_features)
            ])

            return preprocessor

        except Exception as e:
            logging.error("Failed to build preprocessing pipeline.")
            raise CustomException(e, sys)

    def preprocess_data(self, df):
        """Removes unnecessary columns and filters only expense records"""
        df = df[df['Income/Expense'] == 'Expense']
        df = df.drop(columns=['Date', 'Note', 'Income/Expense', 'Currency'], errors='ignore')
        # df['Category'] = df['Category'].str.lower().str.strip()

        for col in ['Mode', 'Category', 'Subcategory']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
        return df

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading train and test CSV files.")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Preprocess raw data
            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            train_df = SubcategoryImputer().fit_transform(train_df)
            test_df = SubcategoryImputer().transform(test_df)

            # Manually clean target column ('Category')
            for df_ in [train_df, test_df]:
                df_['Category'] = df_['Category'].replace(
                    to_replace=r'Equity Mutual Fund [A-Z]',
                    value='Mutual Funds',
                    regex=True
                )

            # Target column
            target_column = 'Category'

            # Separate input features and targets
            input_features_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]
            input_features_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]

            # Handle rare target classes
            logging.info("Original target class distribution (train):")
            logging.info(f"\n{target_feature_train_df.value_counts()}")

            value_counts = target_feature_train_df.value_counts()
            rare_classes = value_counts[value_counts < 18].index.tolist()
            logging.info(f"Rare target classes (<10 occurrences): {rare_classes}")

            target_feature_train_df = target_feature_train_df.apply(
                lambda x: 'other' if x in rare_classes else x
            )

            valid_classes = set(target_feature_train_df.unique())
            target_feature_test_df = target_feature_test_df.apply(
                lambda x: x if x in valid_classes else 'other'
            )

            logging.info("Transformed target class distribution:")
            logging.info(f"\n{target_feature_train_df.value_counts()}")

            # Encode target labels
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(target_feature_train_df)
            y_test = np.array([
                label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1
                for x in target_feature_test_df
            ])

            # Preprocessing pipeline
            preprocessing_obj = self.get_transformer_object()

            logging.info("Fitting and transforming train features.")
            X_train = preprocessing_obj.fit_transform(input_features_train_df)

            logging.info("Transforming test features.")
            X_test = preprocessing_obj.transform(input_features_test_df)

            # Compute sample weights
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

            # Save preprocessor and label encoder
            logging.info("Saving preprocessing pipeline and label encoder.")
            save_object(self.data_transformation_config.preprocessor_obj_path, preprocessing_obj)
            save_object(self.data_transformation_config.label_encoder_path, label_encoder)

            logging.info("Data transformation successful.")
            return np.c_[X_train, y_train], np.c_[X_test, y_test], sample_weights

        except Exception as e:
            logging.error("Error occurred during data transformation.")
            raise CustomException(e, sys)
