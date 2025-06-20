import sys
import os

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from dataclasses import dataclass

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomException
from logger import logging
from utils import evaluate_model, save_object
from data_transformation import DataTransformation
from data_ingestion import DataIngestion

@dataclass
class ModelTrainerConfig:
    model_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            X_train, y_train = train_arr[:,:-1], train_arr[:, -1]
            X_test, y_test = test_arr[:,:-1], test_arr[:,-1]

            models = {
                'LogisticRegression': LogisticRegression(),
                'DecisionTreeClassifier': DecisionTreeClassifier(),
                'RandomForestClassifier': RandomForestClassifier(),
                'AdaBoostClassifier': AdaBoostClassifier(),
                'GradientBoostingClassifier' : GradientBoostingClassifier(),
                'KNeighborsClassifier': KNeighborsClassifier(),
                'SVC':SVC(),
                'CatBoostClassifier': CatBoostClassifier(task_type='GPU', devices='0', silent=True),
                'XGBClassifier': XGBClassifier(tree_method='gpu_hist',gpu_id= 0,  max_depth=6, max_bin=256)
            }

            params = {
                "LogisticRegression": {
                    'C': [0.01, 0.1, 1, 10],
                    'max_iter': [500, 1000],
                    'penalty': ['l2'],
                    'solver': ['saga'],
                    'multi_class': ['multinomial'],
                },
                "DecisionTreeClassifier": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20, 30],
                    'max_features': ['sqrt', 'log2'],
                    'min_samples_leaf': [1, 2, 4],
                    'min_samples_split': [2, 5, 10],
                    'splitter': ['best', 'random']
                },
                "RandomForestClassifier": {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2'],
                    'bootstrap': [True, False]
                },
                "GradientBoostingClassifier": {
                    'loss': ['log_loss'],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5],
                    'min_samples_split': [2, 5],
                    'max_features': ['sqrt', 'log2'],
                    'subsample': [0.8, 1.0]
                },
                "KNeighborsClassifier": {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
                "SVC": {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                },
                "XGBClassifier": {
                    'learning_rate': [0.01, 0.1],
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0],
                    'min_child_weight': [1, 3]
                },
                "CatBoostClassifier": {
                    'depth': [6, 8],
                    'learning_rate': [0.01, 0.1],
                    'iterations': [100, 200],
                    'l2_leaf_reg': [3, 5],
                    'bagging_temperature': [0.0, 0.5]
                },
                "AdaBoostClassifier": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0],
                    'algorithm': ['SAMME', 'SAMME.R']
                }
            }
            
            best_model_accuracy, best_model_instance , report = evaluate_model(
                X_train,y_train,X_test,y_test,models,params
            )

            if best_model_accuracy < 0.6:
                raise CustomException('No Best Model Found')
            
            best_model = models[best_model_instance]
            save_object(self.model_trainer_config.model_path, best_model)
            
            print(report)

        except Exception as e:
            raise CustomException(e,sys)
        
        return best_model_accuracy
    
if __name__ == "__main__":
    train_path, test_path = DataIngestion().initiate_data_ingestion()
    preprocessor = DataTransformation().get_transformer_object()
    train_arr, test_arr = DataTransformation().initiate_data_transformation(train_path, test_path)
    accuracy = ModelTrainer().initiate_model_training(train_arr, test_arr)