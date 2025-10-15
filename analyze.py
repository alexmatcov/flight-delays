# %%

import glob
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import airportsdata
import dask.dataframe as dd
from dask.diagnostics.progress import ProgressBar

from storage import config

# %%
# Load weather data
try:
    weather_path = config.get_s3_path(f"{config.processed_path}/weather_combined.parquet")
    we = dd.read_parquet(weather_path, storage_options=config.s3fs_storage_options)
except:
    weather_path = config.get_s3_path(f"{config.raw_weather_path}/*.csv")
    we = dd.read_csv(weather_path, storage_options=config.s3fs_storage_options, assume_missing=True)

we["time"] = we["time"].astype("string")
we.head()

# %%
# Load flight data
dtype_spec = {
    "Div1Airport": "str",
    "Div1TailNum": "str",
    "Div2Airport": "str",
    "Div2TailNum": "str",
    "Div3Airport": "str",
    "Div3TailNum": "str",
    "CancellationCode": "str",
    "IATA_Code_Originally_Scheduled_Code_Share_Airline": "str",
    "Originally_Scheduled_Code_Share_Airline": "str",
}

try:
    flight_path = config.get_s3_path(f"{config.processed_path}/flight_data.parquet")
    fl = dd.read_parquet(flight_path, storage_options=config.s3fs_storage_options)
except:
    flight_path = config.get_s3_path(f"{config.raw_flights_path}/flight_data_2018_2024.csv")
    fl = dd.read_csv(flight_path, dtype=dtype_spec, storage_options=config.s3fs_storage_options, assume_missing=True)

# Filter interesting flights
fl = fl[fl["Cancelled"] == False]  # type: ignore
fl = fl[fl["ActualElapsedTime"].notnull()]
fl = fl[
    [
        "FlightDate",
        "DepTime",
        "Origin",
        "DepDelayMinutes",
        "ActualElapsedTime",
        "Dest",
        "ArrDelayMinutes",
    ]
]

# Fix time data
airport_tz = airportsdata.load("IATA")


def fix_time(row):
    flight_date_obj = datetime.strptime(row["FlightDate"], "%Y-%m-%d").date()
    hour = int(row["DepTime"] // 100)
    minute = int(row["DepTime"] % 100)
    tzdb_id = airport_tz[row["Origin"]]["tz"]

    if hour == 24:
        hour = 0
        flight_date_obj += timedelta(days=1)
    naive_dt = datetime.combine(flight_date_obj, time(hour, minute))
    localized_dt = naive_dt.replace(tzinfo=ZoneInfo(tzdb_id))
    return localized_dt.isoformat()


fl["dep"] = fl.apply(fix_time, axis=1, meta=("dep", "str"))
fl["arr"] = fl.apply(
    lambda r: (
        datetime.fromisoformat(r["dep"]) + timedelta(minutes=r["ActualElapsedTime"])
    ).isoformat(),
    axis=1,
    meta=("arr", "str"),
)
fl = fl[
    [
        "Origin",
        "dep",
        "Dest",
        "arr",
        "DepDelayMinutes",
        "ArrDelayMinutes",
    ]
]
fl.head(10)

# %%
# Combine flight data with weather station reading for dep and arr hour


def nearest_hour_weather_time(col: str):
    """Returns function to convert the column
    into the nearest hour in the weather data time string format,
    UTC `YYYY-MM-DD HH:MM:SS`"""

    def f(row) -> str:
        d = datetime.fromisoformat(row[col])
        # Round to nearest hour
        if d.minute >= 30:
            d = d.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            d = d.replace(minute=0, second=0, microsecond=0)

        d_utc = d.utctimetuple()

        return datetime(*d_utc[:6]).strftime("%Y-%m-%d %H:%M:%S")

    return f


# Calculate nearest hour in the weather data string format for both
fl["dep_hour"] = fl.apply(
    nearest_hour_weather_time("dep"), axis=1, meta=("dep_hour", "str")
)
fl["arr_hour"] = fl.apply(
    nearest_hour_weather_time("arr"), axis=1, meta=("arr_hour", "str")
)


# Create single common column to merge on
we["time_airport"] = we["time"] + " " + we["airport"].astype("string")
fl["dep_hour_airport"] = fl["dep_hour"] + " " + fl["Origin"].astype("string")
fl["arr_hour_airport"] = fl["arr_hour"] + " " + fl["Dest"].astype("string")


# Perform merge
fl_we = fl.merge(
    we.rename(columns=lambda x: "dep_" + x),
    left_on="dep_hour_airport",
    right_on="dep_time_airport",
    how="inner",
)
fl_we = fl_we.drop(
    columns=["dep_time_airport", "dep_hour_airport", "dep_hour", "dep_airport"]
)
fl_we = fl_we.merge(
    we.rename(columns=lambda x: "arr_" + x),
    left_on="arr_hour_airport",
    right_on="arr_time_airport",
    how="inner",
)
fl_we = fl_we.drop(
    columns=["arr_time_airport", "arr_hour_airport", "arr_hour", "arr_airport"]
)

print("Performing joins...")
with ProgressBar():
    fl_we.compute()

print(len(fl_we))
fl_we.head()

# %%
output_path = config.get_s3_path(f"{config.processed_path}/weather_delay_merged.parquet")
print(f"Saving to {output_path}")
with ProgressBar():
    fl_we.to_parquet(
        output_path,
        storage_options=config.s3fs_storage_options,
        engine='pyarrow',
        compression='snappy'
    )