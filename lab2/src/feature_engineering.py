import dvc.api
from omegaconf import DictConfig, OmegaConf

from src.dataset.dataset import Dataset
from src.logger import ExecutorLogger


def feature_engineering(cfg: DictConfig,logger:ExecutorLogger) -> None:
    logger.info("Training started")
    logger.info("Pipeline Parameters: \n" f"{OmegaConf.to_yaml(cfg)}")
    
    train_ds = Dataset(cfg['data'],logger=logger)
    train_ds.engineer_features()

if __name__ == "__main__":
    cfg = dvc.api.params_show('../../lab2/params.yaml')
    logger = ExecutorLogger("training")
    logger.info("Paramsters: \n"f"{cfg['data']}")
    feature_engineering(cfg=cfg, logger=logger)
