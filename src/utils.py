import os
import sys
import time
import dill
import numpy as np
from pathlib import Path
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score

# Adding project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomException
from logger import logging


def save_object(file_path, obj):
    """
    Save a Python object to disk using dill serialization.

    Args:
        file_path (str): The full path where the object will be saved.
        obj (object): The Python object to serialize.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

        logging.info(f"Object saved successfully at: {file_path}")

    except Exception as e:
        logging.error("Error occurred while saving object.")
        raise CustomException(e, sys)


from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import time

def evaluate_model(X_train, y_train, X_test, y_test, models: dict, params: dict, sample_weights):
    """
    Trains and evaluates multiple models with GridSearchCV and multiple classification metrics.
    """
    report = {}
    best_model = None
    best_accuracy = 0.0

    for model_name, model_instance in models.items():
        try:
            logging.info(f"Running GridSearchCV for model: {model_name}")
            model_params = params.get(model_name, {})

            gs = GridSearchCV(model_instance, model_params, cv=3, scoring='accuracy', n_jobs=-1)
            gs.fit(X_train, y_train)

            logging.info(f"Best parameters for {model_name}: {gs.best_params_}")
            model_instance.set_params(**gs.best_params_)

            start_time = time.time()
            if model_name in ['KNeighborsClassifier', 'SVC']:
                model_instance.fit(X_train, y_train)
            else:
                model_instance.fit(X_train, y_train, sample_weight=sample_weights)
                
            train_time = time.time() - start_time
            logging.info(f"⏱️ {model_name} training time: {train_time:.2f} seconds")

            y_pred = model_instance.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            cls_report = classification_report(y_test, y_pred, zero_division=0)

            logging.info(f"{model_name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")
            logging.info(f"{model_name} - Confusion Matrix:\n{cm}")
            logging.info(f"{model_name} - Classification Report:\n{cls_report}")

            report[model_name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'confusion_matrix': cm,
                'classification_report': cls_report
            }

            if acc > best_accuracy:
                best_model = model_instance
                best_accuracy = acc

        except Exception as e:
            logging.error(f"Exception while training model: {model_name}")
            raise CustomException(e, sys)

    return best_accuracy, best_model, report
