import datetime
import os

from dotenv import load_dotenv
import duckdb
import pandas as pd
from prefect import flow, task
from prefect.tasks import task_input_hash

from globals import logger


@task(
    name="GetDailyData",
    description="get daily weather data from hourly data as dataframe",
    tags=["Get", "DailyData"],
    cache_key_fn=task_input_hash,
    cache_expiration=datetime.timedelta(minutes=10),
    retry_delay_seconds=30,
    retries=3,
    log_prints=True,
    timeout_seconds=60,
)
def get_daily_data(conn, running_date: str) -> pd.DataFrame:
    """Get the daily data from hourly weather data

    Parameters
    ----------
    conn : MotherDuck Database Connection
    running_dt: str
         string format of pipeline running date
    """
    logger.info(f"Get Daily Weather Data of {running_date}")
    conn.sql("USE weather_data")
    df_daily = conn.sql(f"""
                        SELECT strftime(timestamp, '%Y-%m-%d') AS day_date,
                               AVG(temperature) AS temperature,
                               AVG(relative_humidity) AS relative_humidity,
                               AVG(rain) AS rain,
                               AVG(precipitation) AS precipitation,
                               AVG(cloud_cover) AS cloud_cover,
                               AVG(wind_speed) AS wind_speed
                        FROM weather_data.hourly_weather_data
                        WHERE strftime(timestamp, '%Y-%m-%d') = '{running_date}'
                        GROUP BY day_date
                        """).df()
    return df_daily

@task(
    name="GetDailyDataUptoNow",
    description="get daily weather data from hourly data as dataframe",
    tags=["Get", "DailyData"],
    cache_key_fn=task_input_hash,
    cache_expiration=datetime.timedelta(minutes=10),
    retry_delay_seconds=30,
    retries=3,
    log_prints=True,
    timeout_seconds=60,
)
def get_daily_data_upto_now(conn, running_date: str) -> pd.DataFrame:
    """Get the daily data from hourly weather data

    Parameters
    ----------
    conn : MotherDuck Database Connection
    running_dt: str
         string format of pipeline running date
    """
    logger.info(f"Get Daily Weather Data of {running_date}")
    conn.sql("USE weather_data")
    if running_date is None:
        df_daily = conn.sql("""
                        SELECT strftime(timestamp, '%Y-%m-%d') AS day_date,
                               AVG(temperature) AS temperature,
                               AVG(relative_humidity) AS relative_humidity,
                               AVG(rain) AS rain,
                               AVG(precipitation) AS precipitation,
                               AVG(cloud_cover) AS cloud_cover,
                               AVG(wind_speed) AS wind_speed
                        FROM weather_data.hourly_weather_data
                        GROUP BY day_date
                        """).df()
    else:
        df_daily = conn.sql(f"""
                        SELECT strftime(timestamp, '%Y-%m-%d') AS day_date,
                               AVG(temperature) AS temperature,
                               AVG(relative_humidity) AS relative_humidity,
                               AVG(rain) AS rain,
                               AVG(precipitation) AS precipitation,
                               AVG(cloud_cover) AS cloud_cover,
                               AVG(wind_speed) AS wind_speed
                        FROM weather_data.hourly_weather_data
                        WHERE strftime(timestamp, '%Y-%m-%d') > '{running_date}'
                        GROUP BY day_date
                        """).df()
    return df_daily

@task(
    name="GetLastDayDate",
    description="get last daily weather data from daily data as str",
    tags=["Get", "LastDay"],
    cache_key_fn=task_input_hash,
    cache_expiration=datetime.timedelta(minutes=10),
    retry_delay_seconds=30,
    retries=3,
    log_prints=True,
    timeout_seconds=60,
)
def get_last_day_date(conn) -> str:
    """Get the last daily data from hourly weather data

    Parameters
    ----------
    conn : MotherDuck Database Connection
    """
    logger.info("Get Last Daily Weather Data")
    conn.sql("USE weather_data")
    last_day_date = conn.sql("""
                        SELECT strftime(day_date, '%Y-%m-%d') AS day_date
                        FROM weather_data.daily_weather_data
                        ORDER BY day_date DESC
                        LIMIT 1
                        """).df()
    if len(last_day_date) > 0:
        return last_day_date.iloc[0]['day_date']
    else:
        return None

@task(
    name="CheckIfDailyExists",
    description=(
        "Check the MotherDuck if the Data of the passed date exists, and if it"
        " removes it"
    ),
    tags=["Check", "DailyData", "DB"],
    retry_delay_seconds=30,
    retries=3,
    log_prints=True,
    timeout_seconds=60,
)
def check_if_data_exists(conn, running_date: str) -> None:
    """Check the MotherDuck if the Data of the passed date exists,
      and if it removes it

    Parameters
    ----------
    conn : MotherDuck Database Connection
    running_dt: str
         string format of pipeline running date
    """
    logger.info(f"Check if Data Exist for the current-date: {running_date}")
    conn.sql("USE weather_data")
    daily_df = conn.sql(f"""
                       SELECT * FROM weather_data.daily_weather_data 
                        WHERE strftime(day_date, '%Y-%m-%d') = '{running_date}'
                    """).df()
    logger.info(f"found {len(daily_df)} records")
    if len(daily_df) > 0:
        logger.info("data found, and will be removed safely")
        conn.sql(f"""
                DELETE FROM weather_data.daily_weather_data 
                WHERE strftime(day_date, '%Y-%m-%d') = '{running_date}'
            """)
        logger.info("data removed successfully")
    else:
        logger.info(f"no data exists for {running_date}")


@task(
    name="LoadToDailyMotherDuck",
    description="Load Data into MotherDuck Database",
    tags=["load", "updated-data", "motherduck"],
    retry_delay_seconds=60,
    retries=3,
    log_prints=True,
    timeout_seconds=60,
)
def load_to_motherduck(df: pd.DataFrame, conn) -> None:
    """Load Data into Database

    Parameters
    ----------
    df : pd.DataFrame
        API DataFrame
    conn : motherduck database connection
    """
    conn.sql("USE weather_data")
    conn.sql(
        "INSERT INTO weather_data.daily_weather_data SELECT * FROM df"
    )
    logger.info("Data Insert Successfully into daily_weather_data Table")


@task(
    name="DeleteOutofRangeDailyData",
    description="Delete Out of Range Data to mintain Staorage Space",
    tags=["delete", "out-of-range-data", "motherduck", "storage"],
    retry_delay_seconds=60,
    retries=3,
    log_prints=True,
    timeout_seconds=60,
)
def delete_out_of_range_data(conn, thresh_date: str) -> None:
    logger.info("Deleting Out of Range Data")
    conn.sql("USE weather_data")
    conn.sql(f"""
            DELETE FROM weather_data.daily_weather_data 
            WHERE day_date <= CAST('{thresh_date}' AS DATE)-2000
        """)
    logger.info("Data Deleted Successfully")


@flow(
    name="InferenceDataPreparationFlow",
    description="Prepare data to be daily, and insert it to MotherDuck",
    validate_parameters=True,
    log_prints=True,
)
def data_prep_flow(db_token: str, date: str) -> bool:
    """sub-flow of preparing daily data for inference process

    Parameters
    ----------
    db_token: MotherDuck Database Credentials
    date: str
        running date of the process,
        and date used to delete data that exceeds 2000 days from
        this date.
    """
    inference_flag = False
    logger.info("Connecting To MotherDuck to Load Data")
    conn = duckdb.connect(f"md:?motherduck_token={db_token}")
    logger.info("Connection Successfully intiated")
    conn.sql("USE weather_data")
    last_day = get_last_day_date(conn=conn)
    df = get_daily_data_upto_now(conn=conn, running_date=last_day)
    if len(df) > 0:
        inference_flag = True
        logger.info(f"Daily Data for {date} Exists")
        check_if_data_exists(conn=conn, running_date=date)
        load_to_motherduck(df=df, conn=conn)
        delete_out_of_range_data(conn=conn, thresh_date=date)
    logger.info("Connection with MotherDuck Closed")
    return inference_flag

if __name__ == "__main__":
    # Example usage
    load_dotenv()
    date = "2024-04-29"
    data_prep_flow(db_token=os.getenv("MOTHERDUCK_TOKEN"), date=date)