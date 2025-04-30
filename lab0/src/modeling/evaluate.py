import json
import os
import pickle

from skore import EstimatorReport

from src.config import MODELS_DIR, REPORTS_DIR
from src.logger import ExecutorLogger


def evaluate(X_test, y_test, model_name: str, logger:ExecutorLogger) -> None:
    logger.info("loading model")
    with open(os.path.join(MODELS_DIR, model_name, "final_model.pkl"), "rb") as pkl:
        final_model = pickle.load(pkl)
 
    final_report = EstimatorReport(final_model, X_test=X_test, y_test=y_test)
    logger.info("creating evaluation report")
    evaluation_report = {
        "model_name": model_name,
        "estimator_name": final_report.estimator_name_,
        "fitting_time": final_report.fit_time_,
        "accuracy": final_report.metrics.accuracy(),
        "precision": final_report.metrics.precision(),
        "recall": final_report.metrics.recall(),
        "prediction_time": final_report.metrics.timings(),
    }
    logger.info("saving evaluation report")
    if not os.path.exists(os.path.join(REPORTS_DIR, model_name)):
        os.makedirs(os.path.join(REPORTS_DIR, model_name))
    with open(
        os.path.join(REPORTS_DIR, model_name, "evaluation_report.json"), "w"
    ) as js:
        json.dump(evaluation_report, js, indent=4)
