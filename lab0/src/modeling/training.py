import os
import pickle
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from config import MODELS_DIR
from globals import logger


def train(model: BaseEstimator, X_train: np.ndarray, y_train: np.ndarray) -> None:
    """
    Train the model.
    """
    if model is None:
        logger.error("Model is None.")
        raise ValueError("Model is None.")
    model.fit(X_train, y_train)
    logger.success("Model trained.")


def RandomizedSearchCV(
    model: BaseEstimator,
    params: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    """
    Perform Randomized Search CV on the model.
    """
    if model is None:
        logger.error("Model is None.")
        raise ValueError("Model is None.")

    search = RandomizedSearchCV(model, params, n_iter=100, cv=3, verbose=2)
    search.fit(X_train, y_train)

    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best score: {search.best_score_}")

    logger.success("Randomized Search CV completed.")


def save_model(model: None, model_name: str) -> None:
    """
    Save the model.
    """
    if model is None:
        logger.error("Model is None.")
        raise ValueError("Model is None.")

    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    with open(os.path.join(model_path, f"{model_name}.pkl"), "wb") as pkl:
        pickle.dump(model, pkl)

    logger.success(f"Model saved to {os.path.join(MODELS_DIR, model_name)}.")
