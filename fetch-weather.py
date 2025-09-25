# %%
import pandas as pd

airports_csv = "data/airports.csv"
airports_csv_url = "https://davidmegginson.github.io/ourairports-data/airports.csv"

ap_cds: pd.DataFrame
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
ap_cds.head()

# %%
# Download historical weather data for airport locations
# TODO
