import os

import dvc.api
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.logger import ExecutorLogger
from src.modeling.evaluate import evaluate
from src.modeling.training import save_model


def training(cfg: DictConfig,logger:ExecutorLogger) -> None:
    logger.info("Training started")
    logger.info("Pipeline Parameters: \n" f"{OmegaConf.to_yaml(cfg)}")
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    val_df = pd.read_csv(os.path.join(cfg["data"]["processed_data_path"],f"{cfg['data']['val_name']}"))
    X_val = val_df.drop(columns=[cfg['data']['target_col']])
    y_val = val_df[cfg['data']['target_col']]
    evaluate(X_val, y_val, cfg=cfg["evaluate"], logger=logger)
    save_model(model,cfg['model']['model_name'],cfg['model']['model_path'],logger)
    
if __name__ == "__main__":
    cfg = dvc.api.params_show('../../lab2/params.yaml')
    print(cfg)
    logger = ExecutorLogger("training")
    logger.info("Paramsters: \n"f"{cfg['data']}")
    training(cfg=cfg, logger=logger)
