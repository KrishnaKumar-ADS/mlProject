import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object   # make sure utils.py defines save_object(file_path, obj)


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransforation:
    """
    Builds and applies preprocessing for the StudentsPerformance dataset.
    NOTE: Class name kept as 'DataTransforation' to match your existing imports.
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

        # Expected columns (exactly as in the CSV)
        self.num_cols = ["writing score", "reading score"]
        self.cat_cols = [
            "gender",
            "race/ethnicity",
            "parental level of education",
            "lunch",
            "test preparation course",
        ]
        self.target_col = "math score"

    def _validate_columns(self, df: pd.DataFrame, where: str):
        """Validate that required columns exist; raise clear error otherwise."""
        missing = [c for c in (self.num_cols + self.cat_cols + [self.target_col]) if c not in df.columns]
        if missing:
            raise CustomException(
                Exception(
                    f"Missing columns in {where}: {missing}\n"
                    f"Available columns: {df.columns.tolist()}"
                ),
                sys
            )

    def get_data_transformer_object(self):
        """
        Create the preprocessing object:
        - Numeric: impute median, scale (mean/std)
        - Categorical: impute most_frequent, OneHot (no scaling)
        """
        try:
            # Numeric pipeline (dense)
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])

            # Categorical pipeline: OneHot (sparse). No StandardScaler here to avoid sparse-centering error.
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
            ])

            logging.info(f"Numerical columns: {self.num_cols}")
            logging.info(f"Categorical columns: {self.cat_cols}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, self.num_cols),
                    ("cat_pipeline", cat_pipeline, self.cat_cols),
                ],
                remainder="drop",
                sparse_threshold=0.3,  # keep result sparse if mostly sparse
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Read train/test, validate schema, fit/transform, and persist the preprocessor.
        Returns: (train_array, test_array, preprocessor_path)
        """
        try:
            # Read
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train columns: {train_df.columns.tolist()}")
            logging.info(f"Test columns:  {test_df.columns.tolist()}")

            # Validate expected columns early (prevents cryptic KeyErrors later)
            self._validate_columns(train_df, "train_df")
            self._validate_columns(test_df, "test_df")

            # Split into features/target
            input_feature_train_df = train_df.drop(columns=[self.target_col], axis=1)
            target_feature_train_df = train_df[self.target_col]

            input_feature_test_df = test_df.drop(columns=[self.target_col], axis=1)
            target_feature_test_df = test_df[self.target_col]

            # Build preprocessor
            logging.info("Obtaining preprocessing object")
            preprocessor_obj = self.get_data_transformer_object()

            # Fit on train, transform both
            logging.info("Applying preprocessing object on training and testing dataframes")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # Concatenate features with target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Persist preprocessor
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info(f"Saved preprocessing object to {self.data_transformation_config.preprocessor_obj_file_path}")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
