import datetime
import os

from dotenv import load_dotenv
import duckdb
import pandas as pd
from prefect import flow, task
from prefect.tasks import task_input_hash
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.fbprophet import Prophet

from globals import logger


@task(
    name="GetInferenceData",
    description="get daily weather data that are needed to forecast weather",
    tags=["Get", "InferenceData"],
    cache_key_fn=task_input_hash,
    cache_expiration=datetime.timedelta(minutes=10),
    retry_delay_seconds=30,
    retries=3,
    log_prints=True,
    timeout_seconds=60,
)
def get_inference_data(conn, running_date: str) -> pd.DataFrame:
    """connect to motherduck,
    and get data needed to forecast next 30 days

    Parameters
    ----------
    conn : MotherDuck Database Connection
    running_dt: str
         string format of pipeline running date

    Returns
    -------
    pd.DataFrame
        dataframe of daily historical weather data.
    """
    df = conn.sql(
        "SELECT day_date, temperature FROM weather_data.daily_weather_data WHERE"
        f" day_date >= CAST('{running_date}' AS DATE) - INTERVAL '400 days'"
    ).df()
    df.set_index("day_date", inplace=True)
    df.sort_index(inplace=True)
    return df


@task(
    name="ForecastWeather",
    description="Forecast the next 30",
    tags=["Forecast", "Inference"],
    retry_delay_seconds=30,
    retries=3,
    log_prints=True,
    timeout_seconds=30,
)
def forecast_weather(hist_df: pd.DataFrame, running_date: str) -> pd.DataFrame:
    """forecast the next 30 days of weather

    Parameters
    ----------
    hist_df : pd.DataFrame
        dataframe of historical weather data
    running_date : str
        string format of pipeline running date

    Returns
    -------
    pd.DataFrame
        dataframe of forecasted temperature
    """
    try:
        model = Prophet(
            n_changepoints=50,
            changepoint_range=0.95,
            seasonality_prior_scale = 1,
            changepoint_prior_scale = 0.5,
            seasonality_mode="multiplicative",
        )
        model.fit(hist_df)
        future = ForecastingHorizon(list(range(1, 11)), is_relative=True)
        forecast = model.predict(future)
        print(forecast.head())
        preds_df = pd.DataFrame(forecast, columns=["temperature"]).reset_index()
        # Rename the columns
        preds_df.rename(columns={"index": "day_date","temperature":"forecasted_temperature"}, inplace=True)
        print(preds_df.head())
        return preds_df
    except Exception as e:
        logger.error(f"An error occurred during forecasting: {e}")
        raise e


@task(
    name="LoadForecastsIntoMotherDuck",
    description="Load Weather Forecasts into database",
    tags=["Load", "ForecastedData", "Database"],
    retry_delay_seconds=30,
    retries=3,
    log_prints=True,
    timeout_seconds=60,
)
def load_forecasts_into_db(conn, preds_df: pd.DataFrame) -> None:
    """load forecasted temperature to motherduck

    Parameters
    ----------
    conn : MotherDuck Database Connection
    preds_df : pd.DataFrame
        dataframe of forecasted temperature
    """
    conn.sql("INSERT INTO weather_data.predictions SELECT * FROM preds_df")


def delete_out_of_range_data(conn, thresh_date: str) -> None:
    logger.info("Deleting Out of Range Data")
    conn.sql(f"""
            DELETE FROM weather_data.predictions
            WHERE day_date <= CAST('{thresh_date}' AS DATE)-1000
        """)
    logger.info("Data Deleted Successfully")


@flow(
    name="WeatherForecastingFlow",
    description="Forecast Weather, and insert results into DB",
    validate_parameters=True,
    log_prints=True,
)
def forecast_flow(db_token: str, date: str) -> None:
    """flow of inference

    Parameters
    ----------
    db_token : str
        MotherDuck Database Credentials
    date : str
        running date of the process
    model_path : str
        path of pickle file
    """
    logger.info("Connecting To MotherDuck to Get/Load Data")
    with duckdb.connect(f"md:?motherduck_token={db_token}") as conn:
        logger.info("Getting Scoring Data From MotherDuck")
        df = get_inference_data(conn=conn, running_date=date)
        if len(df) > 0:
            preds = forecast_weather(hist_df=df, running_date=date)
            logger.info(f"Model Forecasted Next {len(preds)} days")
            load_forecasts_into_db(conn=conn, preds_df=preds)
            logger.info("Data Loaded into MotherDuck")
            delete_out_of_range_data(conn=conn, thresh_date=date)
        else:
            logger.info("No Records in Scoring data..")
    logger.info("Connection with MotherDuck Closed")


if __name__ == "__main__":
    # Example usage
    load_dotenv()
    date = "2023-10-01"
    forecast_flow(db_token=os.environ.get("MOTHERDUCK_TOKEN", None), date=date)
