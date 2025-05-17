import datetime
import os

from dotenv import load_dotenv
import duckdb
import pandas as pd
from prefect import flow, task
from prefect.tasks import task_input_hash
import requests

from data_models import APIData, URLParams
from globals import logger


@task(
    name="GetAPIData",
    cache_key_fn=task_input_hash,
    cache_expiration=datetime.timedelta(minutes=10),
    retries=3,
    retry_delay_seconds=120,
    timeout_seconds=60,
    log_prints=True,
)
def get_api_data(data_url: str, params: URLParams) -> APIData:
    logger.info("Retrieving weather data from API...")
    response = requests.get(url=data_url, params=params)
    if response.status_code == 200:
        logger.info(f"Data retrieved successfully (status code: {response.status_code})")
        data = response.json()
        required_keys = [
            "time",
            "temperature_2m",
            "relative_humidity_2m",
            "rain",
            "precipitation",
            "cloud_cover",
            "wind_speed_10m",
        ]
        if not all(key in data["hourly"] for key in required_keys):
            raise KeyError("Missing required keys in the API response.")
        for key in required_keys:
            if len(data["hourly"][key]) >= 24 and None not in data["hourly"][key]:
                logger.info("Data is complete and valid.")
            else:
                raise ValueError(f"Missing values in {key} data.")
        return data
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
def transform_api_data(data: APIData) -> pd.DataFrame:
    logger.info("Transforming API data into structured DataFrame...")
    timestamps = data["hourly"]["time"]
    temperatures = data["hourly"]["temperature_2m"]
    relative_humidity = data["hourly"]["relative_humidity_2m"]
    rain = data["hourly"]["rain"]
    precipitation = data["hourly"]["precipitation"]
    cloud_cover = data["hourly"]["cloud_cover"]
    wind_speed = data["hourly"]["wind_speed_10m"]

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "temperature": temperatures,
            "relative_humidity": relative_humidity,
            "rain": rain,
            "precipitation": precipitation,
            "cloud_cover": cloud_cover,
            "wind_speed": wind_speed,
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
def check_if_data_exists(db_conn, running_dt: str) -> None:
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
    db_conn.sql("USE weather_data")
    query = f"""
        SELECT * FROM weather_data.hourly_weather_data 
        WHERE strftime(timestamp, '%Y-%m-%d') = '{running_dt}'
    """
    result_df = db_conn.sql(query).df()
    logger.info(f"Found {len(result_df)} existing records.")

    if not result_df.empty:
        logger.info("Data exists. Deleting previous records...")
        db_conn.sql(f"""
            DELETE FROM weather_data.hourly_weather_data 
            WHERE strftime(timestamp, '%Y-%m-%d') = '{running_dt}'
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
def load_to_motherduck(df: pd.DataFrame, db_conn) -> None:
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
    db_conn.sql("USE weather_data")
    db_conn.sql("INSERT INTO weather_data.hourly_weather_data SELECT * FROM df")
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
def delete_out_of_range_data(db_conn, thresh_dt: str) -> None:
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
    db_conn.sql("USE weather_data")
    db_conn.sql(f"""
        DELETE FROM weather_data.hourly_weather_data 
        WHERE timestamp <= CAST('{thresh_dt}' AS DATE) - 800
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
    api_data = get_api_data(data_url=api_data_url, params=url_params)
    transformed_df = transform_api_data(data=api_data)

    logger.info("Establishing connection to MotherDuck...")
    conn = duckdb.connect(f"md:?motherduck_token={db_token}")
    if conn is None:
        logger.error("Failed to connect to MotherDuck.")
        raise ConnectionError("Connection to MotherDuck failed.")
    if conn:
        logger.info("Connection to MotherDuck established successfully.")
    logger.info("Connected to MotherDuck successfully.")
    check_if_data_exists(db_conn=conn, running_dt=deleting_thresh_dt)
    load_to_motherduck(df=transformed_df, db_conn=conn)
    delete_out_of_range_data(db_conn=conn, thresh_dt=deleting_thresh_dt)

    logger.info("Pipeline completed and connection closed.")


if __name__ == "__main__":
    load_dotenv()
    data_flow(
        api_data_url="https://archive-api.open-meteo.com/v1/archive",
        url_params=URLParams(
            latitude=27,
            longitude=30,
            start_date="2025-04-29",
            end_date="2025-04-30",
            hourly=[
                "temperature_2m",
                "relative_humidity_2m",
                "rain",
                "precipitation",
                "cloud_cover",
                "wind_speed_10m",
            ],
            timezone="Africa/Cairo",
        ),
        db_token=os.getenv("MOTHERDUCK_TOKEN"),
        deleting_thresh_dt="2023-10-01",
        logger=logger,
    )
