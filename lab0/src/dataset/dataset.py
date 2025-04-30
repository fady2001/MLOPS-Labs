import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.dataset.features import extract_features
from src.dataset.preprocess import preprocessor
from src.logger import ExecutorLogger


class Dataset:
    def __init__(self, filename: str, id_col: str, target_col:str,logger: ExecutorLogger):
        """
        Initialize the Dataset class.
        """
        self.filename:str = filename
        self.id_col:str = id_col
        self.target_col:str = target_col
        self.logger:ExecutorLogger = logger
        self.df:pd.DataFrame = self.load_dataset(filename, id_col, logger)
        
    def load_dataset(self,filename: str, id_col:str, logger:ExecutorLogger) -> pd.DataFrame:
        """
        Read a dataset from a CSV file.
        """
        filepath = RAW_DATA_DIR / filename
        
        if not filepath.exists() or not filepath.is_file():
            logger.error(f"File {filepath} does not exist or is not a file.")
            raise FileNotFoundError(f"File {filepath} does not exist or is not a file.")
        df = pd.read_csv(filepath, sep=",")
        if id_col not in df.columns:
            logger.error(f"Column {id_col} not found in the dataset.")
            raise ValueError(f"Column {id_col} not found in the dataset.")
        df.set_index(id_col, inplace=True)
        logger.success(f"Dataset loaded from {filepath}.")
        return df

    @staticmethod
    def split_dataset(df: pd.DataFrame, logger: ExecutorLogger, train_size: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split a dataset into training and testing sets.
        """
        if not 0 < train_size < 1:
            logger.error("train_size must be between 0 and 1.")
            raise ValueError("train_size must be between 0 and 1.")
        if df.empty:
            logger.error("train_size must be between 0 and 1.")
            raise ValueError("DataFrame is empty.")

        train_df, test_df = train_test_split(df, train_size=train_size, random_state=42)
        logger.success(f"Dataset split into training and testing sets with train size {train_size}.")
        logger.info(f"Training set size: {len(train_df)}")
        logger.info(f"Testing set size: {len(test_df)}")
        return train_df, test_df
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Perform feature engineering on the dataset.
        """
        self.df = extract_features(self.df)
        self.save_dataset(self.df, filename=INTERIM_DATA_DIR / self.filename)
        return self.df
    
    def preprocess_dataset(self) -> pd.DataFrame:
        preprocessed_df,encoder,scaler = preprocessor(self.df, output_path=PROCESSED_DATA_DIR / self.filename)
        self.save_dataset(preprocessed_df, filename="train_preprocessed.csv", dir=PROCESSED_DATA_DIR)
        
        
    def save_dataset(self, df: pd.DataFrame, filename: str, dir:str=PROCESSED_DATA_DIR) -> None:
        """
        Save a DataFrame to a CSV file.
        """
        filepath = dir / filename
        df.to_csv(filepath, sep=",")
        self.logger.success(f"Dataset saved to {filepath}.")
        self.logger.info(f"Dataset shape: {df.shape}")