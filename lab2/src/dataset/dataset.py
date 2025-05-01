import os
import pickle
from typing import Tuple

from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import train_test_split

from src.dataset.features import extract_features
from src.dataset.preprocess import after_split_preprocess, before_split_preprocess, process_test
from src.logger import ExecutorLogger


class Dataset:
    def __init__(self, cfg: DictConfig,logger: ExecutorLogger,encoder = None, scaler = None,path=None) -> None:
        """
        Initialize the Dataset class.
        """
        self.cfg:DictConfig = cfg
        self.logger:ExecutorLogger = logger
        self.df:pd.DataFrame = self.load_dataset(cfg["file_name"], cfg["id_col"],path)
        self.encoders = encoder
        self.scalers = scaler
        self.test_id = self.df[cfg["id_col"]]
        
    def load_dataset(self,filename: str, id_col:str, path) -> pd.DataFrame:
        """
        Read a dataset from a CSV file.
        """
        file_path = ""
        if path is None:
            file_path = os.path.join(self.cfg["raw_data_path"],filename)
        else:
            file_path=path
        if not os.path.exists(file_path):
            self.logger.error(f"File {file_path} not found.")
            raise FileNotFoundError(f"File {file_path} not found.")
        df = pd.read_csv(file_path, sep=",")
        if id_col not in df.columns:
            self.logger.error(f"Column {id_col} not found in the dataset.")
            raise ValueError(f"Column {id_col} not found in the dataset.")
        self.logger.success(f"Dataset loaded from {file_path}.")
        return df

    def split_dataset(self,df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split a dataset into training and testing sets.
        """
        train_df, test_df = train_test_split(df, test_size=test_ratio, random_state=self.cfg["random_state"])
        self.logger.success(f"Dataset split into training and testing sets with test size {test_ratio}.")
        self.logger.info(f"Training set size: {len(train_df)}")
        self.logger.info(f"Testing set size: {len(test_df)}")
        return train_df, test_df
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Perform feature engineering on the dataset.
        """
        self.df = extract_features(self.df)
        self.save_dataset(self.df, filename=self.cfg["file_name"], dir=self.cfg["interim_data_path"])
        return self.df
    
    def preprocess_train(self) -> None:
        # drop column with more than 100 unique values
        self.df = self.df.loc[:,self.df.nunique() < 15]
        preprocessed_df,self.encoders = before_split_preprocess(self.df)
        self.save_dataset(preprocessed_df, filename=self.cfg['file_name'], dir=self.cfg["processed_data_path"])
        self.train_df, self.val_df = self.split_dataset(preprocessed_df, self.cfg["test_size"])
        preprocessed_train_df,self.scalers = after_split_preprocess(self.train_df, self.cfg["target_col"])
        preprocessed_val_df = process_test(self.val_df,self.cfg['target_col'], self.encoders, self.scalers)
        self.save_dataset(preprocessed_train_df, filename=self.cfg['file_name'], dir=self.cfg["processed_data_path"])
        self.save_dataset(preprocessed_val_df, filename=self.cfg['val_name'], dir=self.cfg["processed_data_path"])
        
    def preprocess_test(self) -> pd.DataFrame:
        """
        Preprocess the test dataset.
        """
        # drop column with more than 100 unique values
        self.df = self.df.loc[:,self.df.nunique() < 15]     
        if self.encoders is None or self.scalers is None:
            self.logger.error("Encoders and scalers are not available. Please preprocess the training dataset first.")
            raise ValueError("Encoders and scalers are not available. Please preprocess the training dataset first.")
        preprocessed_df = process_test(self.df, self.cfg["target_col"],self.encoders, self.scalers)
        self.save_dataset(preprocessed_df, filename=self.cfg["file_name"], dir=self.cfg["processed_data_path"])
        return preprocessed_df
        
    def save_dataset(self, df: pd.DataFrame, filename: str,dir:str) -> None:
        """
        Save a DataFrame to a CSV file.
        """
        filepath = os.path.join(dir,filename)
        df.to_csv(filepath, sep=",")
        self.logger.success(f"Dataset saved to {filepath}.")
        self.logger.info(f"Dataset shape: {df.shape}")
        
    def get_X_train_val_y_train_val(self) -> tuple[pd.DataFrame, pd.Series,pd.DataFrame, pd.Series]:
        """
        Get the features and target variable from the dataset.
        """
        if self.cfg["target_col"] not in self.df.columns:
            self.logger.error(f'Target column {self.cfg["target_col"]} not found in the dataset.')
            raise ValueError(f'Target column {self.cfg["target_col"]} not found in the dataset.')
        X_train = self.train_df.drop(columns=[self.cfg["target_col"]])
        y_train = self.train_df[self.cfg["target_col"]]
        X_val = self.val_df.drop(columns=[self.cfg["target_col"]])
        y_val = self.val_df[self.cfg["target_col"]]
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
        model_path = os.path.join(self.cfg['model_path'], model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        with open(os.path.join(model_path, f"{encoder_name}.pkl"), "wb") as pkl:
            pickle.dump(self.encoders, pkl)
    
    def save_scalers(self,model_name:str) -> None:
        """
        Save the scalers to a file.
        """
        model_path = os.path.join(self.cfg["model_path"], model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        with open(os.path.join(model_path, "scaler.pkl"), "wb") as pkl:
            pickle.dump(self.scalers, pkl)