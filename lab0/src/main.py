import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from config import MODELS_DIR, PIPELINE_CONFIG
from dataset.dataset import Dataset
from globals import logger
from modeling.evaluate import evaluate
from modeling.training import train
from preprocessing import preprocess_train
from saver import Saver


def main() -> None:
    logger.info("Training started")
    train_ds = Dataset(data="train.csv", target_col="Survived")
    train_ds = train_ds.engineer_features()

    X_train, y_train, X_val, y_val, preprocessor = preprocess_train(
        train_ds,
        pipeline_config=PIPELINE_CONFIG,
    )

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

    train(model, X_train, y_train)

    Saver.save_processed_data(
        X_train, y_train, target_col="Survived", processor=preprocessor, filename="train.csv"
    )
    Saver.save_processed_data(
        X_val, y_val, target_col="Survived", processor=preprocessor, filename="val.csv"
    )

    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor.get_pipeline()),
            ("model", model),
        ]
    )

    Saver.save_model(
        full_pipeline,
        model_name="random_forest",
        dir=MODELS_DIR / "random_forest",
    )

    logger.info("Training finished")
    evaluate(X_val, y_val, model_name="random_forest", logger=logger)

    # test_df = Dataset(
    #     filename="test.csv",
    #     id_col="PassengerId",
    #     target_col=None,
    #     logger=logger,
    #     encoder=train_ds.encoders,
    #     scaler=train_ds.scalers,
    # )
    # test_df = test_df.engineer_features()

    # test_df.preprocess_test()
    # X_test, test_id = test_df.get_test()
    # generate_submission_file("random_forest", X_test, test_id, logger=logger)


if __name__ == "__main__":
    main()
