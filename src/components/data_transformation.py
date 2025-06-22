import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import pandas as pd
from exception import CustomException
from logger import logging
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from utils import save_object


class SubcategoryImputer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to impute missing Subcategory values with corresponding Category.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'Subcategory' in X.columns and 'Category' in X.columns:
            X['Subcategory'] = X['Subcategory'].fillna(X['Category'])
        return X

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=10):
        self.threshold = threshold
        self.rare_classes_ = {}

    def fit(self, X, y=None):
        for col in X.columns:
            value_counts = X[col].value_counts()
            self.rare_classes_[col] = value_counts[value_counts < self.threshold].index.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            X[col] = X[col].apply(lambda x: 'Other' if x in self.rare_classes_[col] else x)
        return X


@dataclass
class DataTransformationConfig:
    """
    Configuration class for transformation artifacts.
    """
    preprocessor_obj_path = os.path.join('artifacts','preprocessor.pkl')
    label_encoder_path = os.path.join('artifacts','label_encoder.pkl')


class DataTransformation:
    """
    Performs transformation of raw input data into trainable numerical format.
    Includes preprocessing pipelines and label encoding.
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformer_object(self):
        """
        Returns ColumnTransformer object with numerical and categorical pipelines.
        """
        try:
            numerical_features = ['Amount']
            categorical_features = ['Mode', 'Subcategory']

            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ('rare_group', RareCategoryGrouper(threshold=10)),
                ('imputer', SubcategoryImputer()),
                ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_features),
                ('cat_pipeline', cat_pipeline, categorical_features)
            ])

            return preprocessor

        except Exception as e:
            logging.error("Failed to build preprocessing pipeline.")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Loads datasets, applies transformations, encodes targets, and returns transformed arrays.
        """
        try:
            logging.info("Reading train and test CSV files.")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Generating preprocessing pipeline.")
            preprocessing_obj = self.get_transformer_object()

            target_column = 'Category'
            input_features_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]

            input_features_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]

            logging.info("Group rare target classes")
            # Group rare target classes
            value_counts = target_feature_train_df.value_counts()
            rare_classes = value_counts[value_counts < 10].index
            target_feature_train_df = target_feature_train_df.apply(lambda x: 'Other' if x in rare_classes else x)
            
            valid_classes = set(target_feature_train_df.unique())
            target_feature_test_df = target_feature_test_df.apply(lambda x: x if x in valid_classes else 'Other')

            logging.info("Encoding target labels.")
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(target_feature_train_df)
            y_test = label_encoder.transform(target_feature_test_df)

            logging.info("Fitting and transforming train features.")
            X_train = preprocessing_obj.fit_transform(input_features_train_df)
            logging.info("Transforming test features.")
            X_test = preprocessing_obj.transform(input_features_test_df)

            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

            logging.info("Saving preprocessing pipeline and label encoder.")
            save_object(self.data_transformation_config.preprocessor_obj_path, preprocessing_obj)
            save_object(self.data_transformation_config.label_encoder_path, label_encoder)

            logging.info("Data transformation successful.")
            return np.c_[X_train, y_train], np.c_[X_test, y_test], sample_weights

        except Exception as e:
            logging.error("Error occurred during data transformation.")
            raise CustomException(e, sys)

