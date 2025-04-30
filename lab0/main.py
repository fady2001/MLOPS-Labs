from src.dataset.dataset import Dataset
from src.logger import ExecutorLogger


def main(logger:ExecutorLogger) -> None:
    logger.info("Training started")
    ds = Dataset(
        filename="train.csv",
        id_col="PassengerId",
        target_col="Survived",
        logger=logger
    )
    ds.engineer_features()
    ds.preprocess_dataset()
    logger.info("Training finished")


if __name__ == "__main__":
    logger = ExecutorLogger("training")
    main(logger)
