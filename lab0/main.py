from sklearn.ensemble import RandomForestClassifier

from src.dataset.dataset import Dataset
from src.logger import ExecutorLogger
from src.modeling.evaluate import evaluate, generate_submission_file
from src.modeling.training import save_model, train


def main(logger:ExecutorLogger) -> None:
    logger.info("Training started")
    train_ds = Dataset(
        filename="train.csv",
        id_col="PassengerId",
        target_col="Survived",
        logger=logger
    )
    train_ds.engineer_features()
    train_ds.preprocess_train()
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    X_train, y_train, X_val, y_val = train_ds.get_X_train_val_y_train_val()
    train(model, X_train,y_train, logger)
    save_model(model, model_name="random_forest", logger=logger)
    train_ds.save_encoders("random_forest","label_encoder")
    train_ds.save_scalers("random_forest")
    
    logger.info("Training finished")
    
    evaluate(X_val, y_val, model_name="random_forest", logger=logger)
    
    test_df = Dataset(
        filename="test.csv",
        id_col="PassengerId",
        target_col=None,
        logger=logger,
        encoder=train_ds.encoders,
        scaler=train_ds.scalers
    )
    test_df.engineer_features()
    test_df.preprocess_test()
    X_test, test_id = test_df.get_test()
    generate_submission_file('random_forest',X_test, test_id , logger=logger)

if __name__ == "__main__":
    logger = ExecutorLogger("training")
    main(logger)
