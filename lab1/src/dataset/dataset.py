import os
import pickle
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import INTERIM_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.dataset.features import extract_features
from src.dataset.preprocess import after_split_preprocess, before_split_preprocess, process_test
from src.logger import ExecutorLogger


class Dataset:
    def __init__(self, filename: str, id_col: str, target_col:str,logger: ExecutorLogger,encoder = None, scaler = None) -> None:
        """
        Initialize the Dataset class.
        """
        self.filename:str = filename
        self.id_col:str = id_col
        self.target_col:str = target_col
        self.logger:ExecutorLogger = logger
        self.df:pd.DataFrame = self.load_dataset(filename, id_col, logger)
        self.encoders = encoder
        self.scalers = scaler
        self.test_id = self.df[self.id_col]
        
    def load_dataset(self,filename: str, id_col:str, logger:ExecutorLogger) -> pd.DataFrame:
        """
        Read a dataset from a CSV file.
        """
        filepath = RAW_DATA_DIR / filename
        
        if not filepath.exists() or not filepath.is_file():
            self.logger.error(f"File {filepath} does not exist or is not a file.")
            raise FileNotFoundError(f"File {filepath} does not exist or is not a file.")
        df = pd.read_csv(filepath, sep=",")
        if id_col not in df.columns:
            self.logger.error(f"Column {id_col} not found in the dataset.")
            raise ValueError(f"Column {id_col} not found in the dataset.")
        self.logger.success(f"Dataset loaded from {filepath}.")
        return df

    def split_dataset(self,df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split a dataset into training and testing sets.
        """
        train_df, test_df = train_test_split(df, test_size=test_ratio, random_state=42)
        self.logger.success(f"Dataset split into training and testing sets with test size {test_ratio}.")
        self.logger.info(f"Training set size: {len(train_df)}")
        self.logger.info(f"Testing set size: {len(test_df)}")
        return train_df, test_df
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Perform feature engineering on the dataset.
        """
        self.df = extract_features(self.df)
        self.save_dataset(self.df, filename=INTERIM_DATA_DIR / self.filename)
        return self.df
    
    def preprocess_train(self) -> None:
        # drop column with more than 100 unique values
        self.df = self.df.loc[:,self.df.nunique() < 15]        
        preprocessed_df,self.encoders = before_split_preprocess(self.df)
        self.save_dataset(preprocessed_df, filename="train.csv", dir=PROCESSED_DATA_DIR)
        self.train_df, self.val_df = self.split_dataset(preprocessed_df)
        preprocessed_train_df,self.scalers = after_split_preprocess(self.train_df, self.target_col)
        preprocessed_val_df = process_test(self.val_df,self.target_col, self.encoders, self.scalers)
        self.save_dataset(preprocessed_train_df, filename="train.csv", dir=PROCESSED_DATA_DIR)
        self.save_dataset(preprocessed_val_df, filename="val.csv", dir=PROCESSED_DATA_DIR)
        
    def preprocess_test(self) -> pd.DataFrame:
        """
        Preprocess the test dataset.
        """
        # drop column with more than 100 unique values
        self.df = self.df.loc[:,self.df.nunique() < 15]     
        if self.encoders is None or self.scalers is None:
            self.logger.error("Encoders and scalers are not available. Please preprocess the training dataset first.")
            raise ValueError("Encoders and scalers are not available. Please preprocess the training dataset first.")
        preprocessed_df = process_test(self.df, self.target_col,self.encoders, self.scalers)
        self.save_dataset(preprocessed_df, filename="test.csv", dir=PROCESSED_DATA_DIR)
        return preprocessed_df
        
    def save_dataset(self, df: pd.DataFrame, filename: str, dir:str=PROCESSED_DATA_DIR) -> None:
        """
        Save a DataFrame to a CSV file.
        """
        filepath = dir / filename
        df.to_csv(filepath, sep=",")
        self.logger.success(f"Dataset saved to {filepath}.")
        self.logger.info(f"Dataset shape: {df.shape}")
        
    def get_X_train_val_y_train_val(self) -> tuple[pd.DataFrame, pd.Series,pd.DataFrame, pd.Series]:
        """
        Get the features and target variable from the dataset.
        """
        if self.target_col not in self.df.columns:
            self.logger.error(f"Target column {self.target_col} not found in the dataset.")
            raise ValueError(f"Target column {self.target_col} not found in the dataset.")
        X_train = self.train_df.drop(columns=[self.target_col])
        y_train = self.train_df[self.target_col]
        X_val = self.val_df.drop(columns=[self.target_col])
        y_val = self.val_df[self.target_col]
        return X_train, y_train,X_val, y_val
    
    def get_test(self) -> Tuple[pd.DataFrame,pd.Series]:
        """
        Get the DataFrame.
        """
        return self.df,self.test_id
    
    
    def save_encoders(self,model_name:str,encoder_name:str) -> None:
        """
        Save the encoders to a file.
        """
        model_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        with open(os.path.join(model_path, f"{encoder_name}.pkl"), "wb") as pkl:
            pickle.dump(self.encoders, pkl)
    
    def save_scalers(self,model_name:str) -> None:
        """
        Save the scalers to a file.
        """
        model_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        with open(os.path.join(model_path, "scaler.pkl"), "wb") as pkl:
            pickle.dump(self.scalers, pkl)