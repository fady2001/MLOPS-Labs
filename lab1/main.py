import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier

from src.dataset.dataset import Dataset
from src.logger import ExecutorLogger
from src.modeling.evaluate import evaluate, generate_submission_file
from src.modeling.training import train
from src.utils import save_components


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = ExecutorLogger("training")
    logger.info("Training started")
    logger.info("Pipeline Parameters: \n" f"{OmegaConf.to_yaml(cfg)}")
    train_ds = Dataset(cfg.pipeline.data,logger=logger)
    train_ds.engineer_features()
    train_ds.preprocess_train()
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    X_train, y_train, X_val, y_val = train_ds.get_X_train_val_y_train_val()
    train(model, X_train,y_train, logger)
    evaluate(X_val, y_val, cfg=cfg.pipeline.evaluate, logger=logger)
    save_components(model, model_name=cfg.pipeline.model.model_name, ds=train_ds, encoder_name=cfg.pipeline.save.encoder_name, logger=logger)
    logger.info("Training finished")
    
    test_df = Dataset(
        cfg=cfg.pipeline.test,
        logger=logger,
        encoder=train_ds.encoders,
        scaler=train_ds.scalers
    )
    test_df.engineer_features()
    test_df.preprocess_test()
    X_test, test_id = test_df.get_test()
    generate_submission_file(X_test, test_id, cfg=cfg.pipeline.evaluate, logger=logger)

if __name__ == "__main__":
    main()
