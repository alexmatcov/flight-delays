# %%

import glob
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import airportsdata
import dask.dataframe as dd

datadir = "data"
weather_files = glob.glob(f"{datadir}/hourly_for_airport/*.csv")
dfs = [dd.read_csv(f) for f in weather_files]

# %%
we = dd.concat(dfs, ignore_index=True)

we["time"] = we["time"].astype("string")
we[we["airport"] != "LBB"].head()
we.head()

# %%
fl = dd.read_csv(
    f"{datadir}/flight_data_2018_2024.csv",
    dtype={
        "Div1Airport": "str",
        "Div1TailNum": "str",
        "Div2Airport": "str",
        "Div2TailNum": "str",
        "Div3Airport": "str",
        "Div3TailNum": "str",
        "CancellationCode": "str",
        "IATA_Code_Originally_Scheduled_Code_Share_Airline": "str",
        "Originally_Scheduled_Code_Share_Airline": "str",
    },
)  # dask guesses the dtype wrong
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
# Combine data with weather station reading for dep and arr hour


def nearest_hour_weather(col: str):
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


fl["dep_hour"] = fl.apply(nearest_hour_weather("dep"), axis=1, meta=("dep_hour", "str"))
fl["arr_hour"] = fl.apply(nearest_hour_weather("arr"), axis=1, meta=("arr_hour", "str"))

fl.head()

# %%
we["time_airport"] = we["time"] + " " + we["airport"].astype("string")
print(we.head())
fl["dep_hour_airport"] = fl["dep_hour"] + " " + fl["Origin"].astype("string")
fl.head()

# %%
fl_we = fl.merge(we, left_on="dep_hour_airport", right_on="time_airport", how="left")
print(len(fl_we))
fl_we.head(10)

# %%
fl_we.to_csv(f"{datadir}/weather_delay.csv")
