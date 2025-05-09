import os

import dagshub
from dotenv import load_dotenv
import mlflow

from globals import logger


def setup_mlflow(tracking_uri: str) -> mlflow.client.MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.client.MlflowClient(tracking_uri=tracking_uri)
    logger.info("MLFlow Client Defined and tracking URI Setted Successfully.")
    return client


def setup_dagshub(cfg) -> None:
    dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init(
        repo_owner=os.getenv("DAGSHUB_USERNAME"),
        repo_name=os.getenv("DAGSHUB_REPO_NAME"),
        mlflow=cfg["flags"]["use_mlflow"],
    )


def authenticate(cfg) -> mlflow.client.MlflowClient:
    load_dotenv()
    """
    Authenticate with MLflow and DagsHub.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    setup_dagshub(cfg)
    client = setup_mlflow(tracking_uri)
    return client
