import pandas as pd
import numpy as np
from srs.exceptions import customexception 
from srs.logger import logging  
import sys 
import os
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from srs.utlis import save_object,evaluate_models

@dataclass
class datatransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifact', "preprocessor.pkl")

class datatransformation:
    def __init__(self):
        self.data_transformation_config = datatransformationconfig()

    def get_data_transformer_object(self):
        try:
            numerical_column = ['reading_score', 'writing_score']
            categorical_column = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"numerical features {numerical_column}")
            logging.info(f"categorical features {categorical_column}")
            preprocessing = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_column),
                    ("cat_pipeline", cat_pipeline, categorical_column)
                ]
            )
            return preprocessing
        except Exception as e:
            raise customexception(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("data is read") 
            logging.info("obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "math_score"
            numerical_column = ['reading_score', 'writing_score']
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("applying preprocessing object on test and train dataframe")
            input_feature_train_df_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_df_arr = preprocessing_obj.transform(input_feature_test_df)
            train_arr = np.c_[
                input_feature_train_df_arr, np.array(target_feature_train_df)
            ]    
            test_arr = np.c_[
                input_feature_test_df_arr, np.array(target_feature_test_df)
            ]
            logging.info("save processing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise customexception(e, sys)
