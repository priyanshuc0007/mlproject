import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
     AdaBoostRegressor,
     GradientBoostingRegressor,
     RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from srs.exceptions import customexception
from srs.logger import logging
from srs.utlis import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path= os.path.join("artifact", "trainer.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, test_array, train_array):
        logging.info("splitting training and test data")
        
        try:
            X_train, Y_train, X_test, Y_test = (
                train_array[:, :-1],  # All columns except the last one as features
                train_array[:, -1],   # Last column as the target variable
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "k-Neighbour classifier": KNeighborsRegressor(),
                "XGBoost Classifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor(),
            }
            model_report:dict=evaluate_models(x_train=X_train,y_train=Y_train,x_test=X_test,y_test=Y_test, models=models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise customexception("No best model found")
            logging.info("Best model found on both test and train")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(X_test)

            r2_square = r2_score(Y_test, predicted)
            return r2_square

        except Exception as e:
            raise customexception(e, sys)
