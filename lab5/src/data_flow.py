import datetime

import duckdb
import pandas as pd
from prefect import flow, task
from prefect.tasks import task_input_hash
import requests

from .data_models import APIData, URLParams


@task(
    name="GetAPIData",
    cache_key_fn=task_input_hash,
    cache_expiration=datetime.timedelta(minutes=10),
    retries=3,
    retry_delay_seconds=120,
    timeout_seconds=60,
    log_prints=True,
)
def get_api_data(data_url: str, params: URLParams, logger) -> APIData:
    logger.info("Retrieving weather data from API...")
    response = requests.get(url=data_url, params=params)

    if response.status_code == 200:
        logger.info(f"Data retrieved successfully (status code: {response.status_code})")
        data = response.json()
        timestamps = data.get("hourly", {}).get("time", [])
        temps = data.get("hourly", {}).get("temperature_2m", [])
        humedity = data.get("hourly", {}).get("relative_humidity_2m", [])
        rain = data.get("hourly", {}).get("rain", [])
        precipitation = data.get("hourly", {}).get("precipitation", [])
        cloud_cover = data.get("hourly", {}).get("cloud_cover", [])

        if len(temps) >= 24 and None not in temps:
            logger.info("Data is complete and valid.")
            return data
        else:
            raise ValueError("Retrieved data is incomplete or contains missing values.")
    else:
        raise ConnectionError(f"Failed to retrieve data (status code: {response.status_code})")


@task(
    name="TransformToDataFrame",
    description="Convert API JSON data to DataFrame for loading into MotherDuck",
    tags=["transform", "prepare-api-data"],
    retries=2,
    retry_delay_seconds=30,
    timeout_seconds=60,
    log_prints=True,
)
def transform_api_data(data: APIData, logger) -> pd.DataFrame:
    logger.info("Transforming API data into structured DataFrame...")
    timestamps = data["hourly"]["time"]
    temperatures = data["hourly"]["temperature_2m"]
    timezone = data["timezone"]

    df = pd.DataFrame(
        {
            "location_id": [75354428] * len(timestamps),
            "reading_timestamp": pd.to_datetime(timestamps),
            "temperature": temperatures,
            "tz": [timezone] * len(timestamps),
        }
    )

    logger.info("Transformation complete. Data is ready for loading.")
    return df


@task(
    name="CheckIfDataExist",
    description="Check and delete existing data for a specific date in MotherDuck",
    tags=["quality-check", "motherduck"],
    retries=3,
    retry_delay_seconds=60,
    timeout_seconds=120,
    log_prints=True,
)
def check_if_data_exists(db_conn, running_dt: str, logger) -> None:
    """
    Check if weather data for a specific date exists in the DB, and delete it if found.

    Parameters:
    ----------
    db_conn : duckdb.DuckDBPyConnection
        MotherDuck database connection.
    running_dt : str
        Date to check in YYYY-MM-DD format.
    """
    logger.info(f"Checking for existing data for date: {running_dt}")
    db_conn.sql("USE ml_apps")
    query = f"""
        SELECT * FROM ml_apps.iti_weather_forecasting.hourly_weather_data 
        WHERE strftime(reading_timestamp, '%Y-%m-%d') = '{running_dt}'
    """
    result_df = db_conn.sql(query).df()
    logger.info(f"Found {len(result_df)} existing records.")

    if not result_df.empty:
        logger.info("Data exists. Deleting previous records...")
        db_conn.sql(f"""
            DELETE FROM ml_apps.iti_weather_forecasting.hourly_weather_data 
            WHERE strftime(reading_timestamp, '%Y-%m-%d') = '{running_dt}'
        """)
        logger.info("Old data deleted successfully.")


@task(
    name="LoadToMotherDuck",
    description="Insert transformed weather data into MotherDuck",
    tags=["load", "motherduck"],
    retries=3,
    retry_delay_seconds=60,
    timeout_seconds=120,
    log_prints=True,
)
def load_to_motherduck(df: pd.DataFrame, db_conn, logger) -> None:
    """
    Load a weather DataFrame into the MotherDuck database.

    Parameters:
    ----------
    df : pd.DataFrame
        Transformed weather data.
    db_conn : duckdb.DuckDBPyConnection
        MotherDuck database connection.
    """
    logger.info("Inserting data into MotherDuck...")
    db_conn.sql("USE ml_apps")
    db_conn.sql("INSERT INTO ml_apps.iti_weather_forecasting.hourly_weather_data SELECT * FROM df")
    logger.info("Data inserted successfully.")


@task(
    name="DeleteOutOfRangeData",
    description="Delete records older than 800 days to maintain storage",
    tags=["cleanup", "motherduck", "storage"],
    retries=3,
    retry_delay_seconds=60,
    timeout_seconds=120,
    log_prints=True,
)
def delete_out_of_range_data(db_conn, thresh_dt: str, logger) -> None:
    """
    Delete weather data older than 800 days before the threshold date.

    Parameters:
    ----------
    db_conn : duckdb.DuckDBPyConnection
        MotherDuck database connection.
    thresh_dt : str
        Threshold date in YYYY-MM-DD format.
    """
    logger.info("Cleaning up old data...")
    db_conn.sql("USE ml_apps")
    db_conn.sql(f"""
        DELETE FROM ml_apps.iti_weather_forecasting.hourly_weather_data 
        WHERE reading_timestamp <= CAST('{thresh_dt}' AS DATE) - 800
    """)
    logger.info("Old records deleted successfully.")


@flow(
    name="DataFlow",
    description="Orchestrates fetching, transforming, and loading weather data into MotherDuck.",
    validate_parameters=True,
    log_prints=True,
)
def data_flow(
    api_data_url: str,
    url_params: URLParams,
    db_token: str,
    deleting_thresh_dt: str,
    logger,
) -> None:
    """
    Main data pipeline flow to get weather data and store it in MotherDuck.

    Parameters:
    ----------
    api_data_url : str
        Weather API endpoint.
    url_params : URLParams
        Parameters for the weather API request.
    db_token : str
        MotherDuck access token.
    deleting_thresh_dt : str
        Date used as a threshold for data deletion (YYYY-MM-DD).
    """
    api_data = get_api_data(data_url=api_data_url, params=url_params, logger=logger)
    transformed_df = transform_api_data(data=api_data, logger=logger)

    logger.info("Establishing connection to MotherDuck...")
    with duckdb.connect(f"md:?motherduck_token={db_token}") as conn:
        logger.info("Connected to MotherDuck successfully.")
        check_if_data_exists(db_conn=conn, running_dt=deleting_thresh_dt, logger=logger)
        load_to_motherduck(df=transformed_df, db_conn=conn, logger=logger)
        delete_out_of_range_data(db_conn=conn, thresh_dt=deleting_thresh_dt, logger=logger)

    logger.info("Pipeline completed and connection closed.")
