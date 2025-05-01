import os

import dvc.api
from omegaconf import DictConfig, OmegaConf

from src.dataset.dataset import Dataset
from src.logger import ExecutorLogger


def preprocess(cfg: DictConfig,logger:ExecutorLogger) -> None:
    logger.info("Training started")
    logger.info("Pipeline Parameters: \n" f"{OmegaConf.to_yaml(cfg)}")
    train_ds = Dataset(cfg=cfg['data'],logger=logger,path=os.path.join(cfg["data"]["interim_data_path"],f"{cfg['data']['file_name']}"))
    train_ds.preprocess_train()
    train_ds.save_encoders(cfg["model"]["model_name"],cfg["save"]["encoder_name"])
    train_ds.save_scalers(cfg["model"]["model_name"])
    
if __name__ == "__main__":
    cfg = dvc.api.params_show('../../lab2/params.yaml')
    print(cfg)
    logger = ExecutorLogger("training")
    logger.info("Paramsters: \n"f"{cfg['data']}")
    preprocess(cfg=cfg, logger=logger)
