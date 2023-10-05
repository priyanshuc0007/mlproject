import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from srs.exceptions import customexception  # Ensure this import is correct

def save_object(file_path, obj):
    """
    Save a Python object to a file using dill serialization.
    
    Parameters:
        file_path (str): The path to the file where the object will be saved.
        obj (object): The Python object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj) 

    except Exception as e:
        raise customexception(e, sys)

def evaluate_models(x_train, y_train, x_test, y_test, models):
    """
    Evaluate machine learning models on training and testing data.
    
    Parameters:
        x_train (array-like): Training features.
        y_train (array-like): Training target values.
        x_test (array-like): Testing features.
        y_test (array-like): Testing target values.
        models (dict): A dictionary of model names and corresponding model objects.
        
    Returns:
        dict: A dictionary of model names and their R-squared scores on the testing data.
    """
    try:
        report = {}  # Initialize the report dictionary
        
        for model_name, model in models.items():
            model.fit(x_train, y_train)  # Train model
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score
        
        return report  # Return the report dictionary after evaluating all models
        
    except Exception as e:
        raise customexception(e, sys)
