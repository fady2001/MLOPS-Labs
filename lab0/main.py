from src.dataset import Dataset
from src.logger import ExecutorLogger


def main(logger) -> None:
    logger.info("Training started")
    df = Dataset.load_dataset("train.csv", "PassengerId", logger)
    train,test  = Dataset.split_dataset(df, logger, train_size=0.8)
    logger.info("Training finished")


if __name__ == "__main__":
    logger = ExecutorLogger("training")
    main(logger)
