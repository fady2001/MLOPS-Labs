
from sklearn.base import BaseEstimator

from src.dataset.dataset import Dataset
from src.logger import ExecutorLogger
from src.modeling.training import save_model


def save_components(model:BaseEstimator,model_name:str,ds:Dataset,encoder_name:str,logger:ExecutorLogger) -> None:
    """
    Save an object to a pickle file.
    """
    save_model(model, model_name, logger=logger)
    ds.save_encoders(model_name,encoder_name)
    ds.save_scalers(model_name)