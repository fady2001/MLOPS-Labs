import json
import os
import pickle

from omegaconf import DictConfig
import pandas as pd
from skore import EstimatorReport

from src.logger import ExecutorLogger


def evaluate(X_test:pd.DataFrame, y_test:pd.Series, cfg: DictConfig, logger:ExecutorLogger) -> None:
    logger.info("loading model")
    with open(os.path.join(cfg.model_path, cfg.model_name, f"{cfg.model_name}.pkl"), "rb") as pkl:
        final_model = pickle.load(pkl)
 
    final_report = EstimatorReport(final_model, X_test=X_test, y_test=y_test)
    logger.info("creating evaluation report")
    evaluation_report = {
        "model_name": cfg.model_name,
        "estimator_name": final_report.estimator_name_,
        "fitting_time": final_report.fit_time_,
        "accuracy": final_report.metrics.accuracy(),
        "precision": final_report.metrics.precision(),
        "recall": final_report.metrics.recall(),
        "prediction_time": final_report.metrics.timings(),
    }
    logger.info("saving evaluation report")
    if not os.path.exists(os.path.join(cfg.reports_path, cfg.model_name)):
        os.makedirs(os.path.join(cfg.reports_path, cfg.model_name))
    with open(
        os.path.join(cfg.reports_path, cfg.model_name, "evaluation_report.json"), "w"
    ) as js:
        json.dump(evaluation_report, js, indent=4)
        
def generate_submission_file(X_test:pd.DataFrame,test_id:pd.Series,cfg:DictConfig, logger:ExecutorLogger) -> None:
    logger.info("loading model")
    with open(os.path.join(cfg.model_path, cfg.model_name, f"{cfg.model_name}.pkl"), "rb") as pkl:
        final_model = pickle.load(pkl)
    logger.info("creating submission file")
    submission_df = pd.DataFrame()
    submission_df[cfg.id_col] = test_id
    submission_df[cfg.target_col] = final_model.predict(X_test)
    submission_df.to_csv(os.path.join(cfg.model_path, cfg.model_name, "submission.csv"), index=False)