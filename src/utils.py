import os
import sys
import pandas as pd
import numpy as np
import dill
from pathlib import Path
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import r2_score
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj,file_obj)
        logging.info("Successfully Created Pickle file")

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    report = {}
    best_model = None
    best_accuracy = 0

    for model_name, model_instance in models.items():
        try:
            model_params = params.get(model_name, {})

            print(model_name)
            gs = GridSearchCV(model_instance,model_params, cv = 3, scoring='accuracy', n_jobs=7)
            gs.fit(X_train,y_train)

            # print(**gs.best_params_)
            model_instance.set_params(**gs.best_params_)
            model_instance.fit(X_train,y_train)

            train_model_accuracy = cross_val_score(model_instance, X_train, y_train, cv=5, scoring='accuracy').mean()
            test_model_accuracy = cross_val_score(model_instance, X_test, y_test, cv=5, scoring='accuracy').mean()

            report[model_name] = test_model_accuracy

            if test_model_accuracy > best_accuracy:
                best_model = model_instance
                best_accuracy = test_model_accuracy
        
        except Exception as e:
            raise CustomException(e,sys)
    
    return best_accuracy, best_model, report

