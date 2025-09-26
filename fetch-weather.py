# %%

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from meteostat import Hourly, Stations
from tqdm import tqdm


def uprint(
    *values: object,
    sep: str | None = " ",
    end: str | None = "\n",
    file=None,
    flush=False,
):
    """print if *NOT* running in interactive mode"""
    if not hasattr(sys, "ps1"):
        print(*values, sep=sep, end=end, file=file, flush=flush)


data_dir = Path("data")
airports_csv = data_dir / Path("airports.csv")
airports_csv_url = "https://davidmegginson.github.io/ourairports-data/airports.csv"

try:
    ap_cds = pd.read_csv(airports_csv)
except FileNotFoundError:
    ap_cds = pd.read_csv(airports_csv_url)
    ap_cds.to_csv(airports_csv, index=False)

ap_cds = ap_cds[["type", "iso_country", "latitude_deg", "longitude_deg", "iata_code"]]
ap_cds.head()

# %%
# Cleaning
ok_types = ["small_airport", "closed", "medium_airport", "large_airport"]
ap_cds = ap_cds[ap_cds["type"].isin(ok_types)]
ap_cds = ap_cds[ap_cds["iata_code"].notna()]
ap_cds = ap_cds[ap_cds["iso_country"] == "US"]

ap_cds = ap_cds.drop(columns=["type", "iso_country"])
ap_cds.count()

# %%
# Find station(s) to correspond to each airport
st_for_ap: dict[str, str] = {}
stations = Stations()

# This timestamp will also be used for hourly data fetching
start = pd.Timestamp(datetime(2018, 1, 1))
end = pd.Timestamp(datetime(2024, 12, 31))

uprint("Get locations for weather stations next to each airport")
for ap_t in tqdm(ap_cds.iterrows(), total=len(ap_cds)):
    ap = ap_t[1]

    iata = ap.iata_code
    st = stations.nearby(ap.latitude_deg, ap.longitude_deg).fetch()

    meteostat_id = (
        st[(st["hourly_start"] <= start) & (end <= st["hourly_end"])]
        .head(1)
        .index.tolist()[0]
    )
    st_for_ap[iata] = meteostat_id

st_for_ap  # type: ignore
# %%
uprint("Download hourly weather data of airports")
hourly_data_dir = data_dir / Path("hourly_for_airport")
hourly_data_dir.mkdir(exist_ok=True)
for ap, st in tqdm(list(st_for_ap.items())):
    hr = Hourly(st, start, end).fetch()
    hr["airport"] = ap
    hr.to_csv(hourly_data_dir / Path(f"{ap}.csv"))
