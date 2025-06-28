import sys
import os
from pathlib import Path
from dataclasses import dataclass

import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomException
from logger import logging
from utils import evaluate_model, save_object
from components.data_transformation import DataTransformation
from components.data_ingestion import DataIngestion

@dataclass
class ModelTrainerConfig:
    """
    Configuration class for storing model artifact path.
    """
    model_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    """
    Class for training multiple ML models, evaluating them, and saving the best performer.
    """
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr, sample_weights):
        """
        Train and evaluate multiple models and save the best performing one.
        """
        try:
            logging.info("Splitting features and target from training and testing arrays.")
            X_train, y_train = train_arr[:,:-1], train_arr[:, -1]
            X_test, y_test = test_arr[:,:-1], test_arr[:,-1]

            models = {
                'LogisticRegression': LogisticRegression(),
                'DecisionTreeClassifier': DecisionTreeClassifier(),
                'RandomForestClassifier': RandomForestClassifier(),
                'AdaBoostClassifier': AdaBoostClassifier(),
                'GradientBoostingClassifier' : GradientBoostingClassifier(),
                'KNeighborsClassifier': KNeighborsClassifier(),
                # 'SVC':SVC(),
                'CatBoostClassifier': CatBoostClassifier(silent=True),
                'XGBClassifier': XGBClassifier(tree_method='hist')
            }

            # Define hyperparameter grids for each model
            params = {
                "LogisticRegression": {
                    'C': [0.01, 0.1, 1, 10],
                    'max_iter': [1000, 2000],
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
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'max_features': ['sqrt', 'log2'],
                    'bootstrap': [True, False]
                },

                "GradientBoostingClassifier":{
                    'learning_rate': [0.03], # 0.05
                    'loss': ['log_loss'], 
                    'max_depth': [5], 
                    'max_features': ['log2'], 
                    'min_samples_split': [7], 
                    'n_estimators': [300], #200  
                    'subsample': [0.9]
                    },

                # "GradientBoostingClassifier": {
                #     'loss': ['log_loss'],
                #     'learning_rate': [0.01, 0.05, 0.1],
                #     'n_estimators': [100, 200],
                #     'max_depth': [3, 5],
                #     'min_samples_split': [2, 5, 7],
                #     'max_features': ['sqrt', 'log2'],
                #     'subsample': [0.8, 0.9,  1.0]
                # },
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
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1, 1.0],
                    'algorithm': ['SAMME', 'SAMME.R']
                }
            }


            mlflow.set_experiment('expense-categorizer')

            logging.info("Evaluating all models with grid search and cross-validation.")
            with mlflow.start_run() as run:
                best_model_accuracy, best_model_instance , report = evaluate_model(
                    X_train, y_train, X_test, y_test, models, params, sample_weights=sample_weights
                )

                if best_model_accuracy < 0.6:
                    raise CustomException('No suitable model found with acceptable accuracy.')

                best_model = best_model_instance

                mlflow.log_param("selected_model", best_model.__class__.__name__)
                mlflow.log_metric("accuracy", best_model_accuracy)

                # ✅ Log model correctly
                mlflow.sklearn.log_model(best_model, artifact_path="model")

                # ✅ Get the model URI correctly
                model_uri = f"runs:/{run.info.run_id}/model"

                
                result = mlflow.register_model(model_uri, "ExpenseCategorizerModel")
                print(f"Model registered as version: {result.version}")

                # Save locally as well if needed
                save_object(self.model_trainer_config.model_path, best_model)
                logging.info(f"Best model saved to {self.model_trainer_config.model_path} with accuracy {best_model_accuracy:.2f}")

                print("Model Report:", report)
                return best_model_accuracy

        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    train_path, test_path = DataIngestion().initiate_data_ingestion()
    train_arr, test_arr, sample_weights = DataTransformation().initiate_data_transformation(train_path, test_path)
    accuracy = ModelTrainer().initiate_model_training(train_arr, test_arr, sample_weights=sample_weights)
    print(f"Final selected model accuracy: {accuracy:.2f}")
