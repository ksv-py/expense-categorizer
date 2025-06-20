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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from utils import save_object
from data_ingestion import DataIngestion


class SubcategoryImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self  # Nothing to compute, so return self

    def transform(self, X):
        X = X.copy()
        if 'Subcategory' in X.columns and 'Category' in X.columns:
            X['Subcategory'] = X['Subcategory'].fillna(X['Category'])
        return X


@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformer_object(self):
        try:
            numerical_features = ['Amount']  # only if you plan to use amount
            categorical_features = ['Mode', 'Subcategory']  # 'Category' is the target

            num_pipeline = Pipeline(
                steps=[
                    ('Imputer', SimpleImputer(strategy= 'mean')),
                    ('Standard Scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('Imputer', SubcategoryImputer()),
                    ('One Hot Encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
                    ('Standard Scaler', StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_features),
                    ('cat_pipeline', cat_pipeline, categorical_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessing_obj = self.get_transformer_object()
            target_column = 'Category'

            input_feature_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr

        except Exception as e:
            raise CustomException(e, sys)

