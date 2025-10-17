# %%
# One-time migration of local data to MinIO
# Use this script to upload existing local files to S3 storage
# Normal pipeline writes directly to MinIO (fetch-weather.py, analyze.py)

from pathlib import Path
from storage import config, Storage

# %%
storage = Storage()
storage.wait_for_minio()

# %%
# Upload flight data
data_dir = Path("data")
flight_csv = data_dir / "flight_data_2018_2024.csv"

if flight_csv.exists():
    s3_key = f"{config.raw_flights_path}/flight_data_2018_2024.csv"
    storage.upload_file(str(flight_csv), s3_key)

# %%
# Upload airports reference
airports_csv = data_dir / "airports.csv"

if airports_csv.exists():
    s3_key = f"{config.raw_airports_path}/airports.csv"
    storage.upload_file(str(airports_csv), s3_key)

# %%
# Upload weather data
weather_dir = data_dir / "hourly_for_airport"

if weather_dir.exists():
    storage.upload_directory(
        str(weather_dir),
        config.raw_weather_path,
        pattern="*.csv"
    )