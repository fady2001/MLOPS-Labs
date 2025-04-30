import pandas as pd
from sklearn.model_selection import train_test_split

from config import RAW_DATA_DIR
from logger import ExecutorLogger


class DatasetLoader:
    
    @staticmethod
    def load_dataset(filename: str, id_col:str, logger:ExecutorLogger) -> pd.DataFrame:
        """
        Read a dataset from a CSV file.
        """
        filepath = RAW_DATA_DIR / filename
        
        if not filepath.exists() or not filepath.is_file():
            logger.error(f"File {filepath} does not exist or is not a file.")
            raise FileNotFoundError(f"File {filepath} does not exist or is not a file.")
        df = pd.read_csv(filepath, sep=",", header=0, index_col=0)
        df.columns = df.columns.str.strip()
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