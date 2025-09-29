# %%

import dask.dataframe as dd

datadir = "data"

weather = dd.read_csv(f"{datadir}/hourly_for_airport/*.csv")
print(f"Loaded weather data with shape: {weather.shape}")
print(f"Number of rows: {weather.shape[0]}")
print(f"Columns: {list(weather.columns)}")
print(weather.head())
# %%
flights = dd.read_csv(
    f"{datadir}/flight_data_2018_2024.csv",
    dtype={
        "Div1Airport": "str",
        "Div1TailNum": "str",
        "Div2Airport": "str",
        "Div2TailNum": "str",
        "IATA_Code_Originally_Scheduled_Code_Share_Airline": "str",
        "Originally_Scheduled_Code_Share_Airline": "str",
    },  # dask guesses the dtype wrong
)
print(flights.head())
