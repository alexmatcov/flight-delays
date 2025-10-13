# %%


from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import airportsdata
import dask.dataframe as dd

datadir = "data"

weather = dd.read_csv(f"{datadir}/hourly_for_airport/*.csv")
print(f"Loaded weather data with shape: {weather.shape}")
print(f"Number of rows: {weather.shape[0]}")
print(f"Columns: {list(weather.columns)}")
weather.head()
# %%
flights_all = dd.read_csv(
    f"{datadir}/flight_data_2018_2024.csv",
    dtype={
        "Div1Airport": "str",
        "Div1TailNum": "str",
        "Div2Airport": "str",
        "Div2TailNum": "str",
        "IATA_Code_Originally_Scheduled_Code_Share_Airline": "str",
        "Originally_Scheduled_Code_Share_Airline": "str",
    },
)  # dask guesses the dtype wrong
flights_raw = flights_all[flights_all["Cancelled"] == False]  # type: ignore
flights_raw = flights_raw[
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


flights_raw["dep"] = flights_raw.apply(fix_time, axis=1, meta=("dep", "str"))
flights_raw["arr"] = flights_raw.apply(
    lambda r: (
        datetime.fromisoformat(r["dep"]) + timedelta(minutes=r["ActualElapsedTime"])
    ).isoformat(),
    axis=1,
    meta=("arr", "str"),
)
flights = flights_raw[
    [
        "Origin",
        "dep",
        "Dest",
        "arr",
        "DepDelayMinutes",
        "ArrDelayMinutes",
    ]
]
flights.head(10)
