#!/usr/bin/env python3
"""
Download data from MinIO storage to local filesystem
"""

from storage import (
    config,
    download_file,
    download_directory,
    load_dataframe,
    list_files,
    wait_for_minio
)


def download_raw_flights(output_dir: str = "data/downloaded"):
    """Download raw flight data from MinIO"""
    print("\n=== Downloading Flight Data ===")
    count = download_directory(
        config.raw_flights_path,
        f"{output_dir}/flights"
    )
    return count


def download_raw_weather(output_dir: str = "data/downloaded"):
    """Download raw weather data from MinIO"""
    print("\n=== Downloading Weather Data ===")
    count = download_directory(
        config.raw_weather_path,
        f"{output_dir}/weather"
    )
    return count


def download_processed_data(output_dir: str = "data/downloaded"):
    """Download processed data from MinIO"""
    print("\n=== Downloading Processed Data ===")
    count = download_directory(
        config.processed_path,
        f"{output_dir}/processed"
    )
    return count


def load_processed_parquet():
    """Load processed data directly into memory as DataFrame"""
    print("\n=== Loading Processed Parquet ===")
    
    # Try different processed parquet files
    parquet_files = [
        "flight_data.parquet",
        "weather_combined.parquet", 
        "weather_delay_merged.parquet",
        "weather_delay_combined.parquet"
    ]
    
    for filename in parquet_files:
        path = f"{config.processed_path}/{filename}"
        print(f"Trying: {path}")
        df = load_dataframe(path, format='parquet')
        if df is not None:
            return df
    
    print("⚠ No processed parquet files found")
    return None


def show_storage_contents():
    """Display all files in MinIO storage"""
    print("\n=== Storage Contents ===")
    
    sections = [
        ("Raw Flights", config.raw_flights_path),
        ("Raw Weather", config.raw_weather_path),
        ("Raw Airports", config.raw_airports_path),
        ("Processed", config.processed_path),
        ("Results", config.results_path)
    ]
    
    for name, prefix in sections:
        files = list_files(prefix)
        print(f"\n{name} ({len(files)} files):")
        
        # Separate by format
        csv_files = [f for f in files if f.endswith('.csv')]
        parquet_files = [f for f in files if f.endswith('.parquet')]
        part_files = [f for f in files if f.endswith('.part')]
        
        if csv_files:
            print(f"  CSV: {len(csv_files)} files")
            for f in csv_files[:3]:
                print(f"    - {f}")
            if len(csv_files) > 3:
                print(f"    ... and {len(csv_files) - 3} more")
        
        if parquet_files:
            print(f"  Parquet: {len(parquet_files)} files")
            for f in parquet_files[:3]:
                print(f"    - {f}")
            if len(parquet_files) > 3:
                print(f"    ... and {len(parquet_files) - 3} more")
        
        if part_files:
            print(f"  Partitions: {len(part_files)} files")


def download_processed_parquet(output_dir: str = "data/downloaded"):
    """Download processed parquet files (recommended for analytics)"""
    print("\n=== Downloading Processed Parquet Files ===")
    
    from pathlib import Path
    output_path = Path(output_dir) / "processed"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all parquet files from processed/
    all_files = list_files(config.processed_path)
    parquet_files = [f for f in all_files if f.endswith('.parquet')]
    
    if not parquet_files:
        print("⚠ No parquet files found in processed/")
        return 0
    
    print(f"Found {len(parquet_files)} parquet files")
    
    downloaded = 0
    for s3_key in parquet_files:
        filename = Path(s3_key).name
        local_path = output_path / filename
        if download_file(s3_key, str(local_path), show_progress=True):
            downloaded += 1
    
    print(f"✓ Downloaded {downloaded} parquet files to {output_path}")
    return downloaded


def main():
    """Download all data from MinIO"""
    wait_for_minio()
    
    show_storage_contents()
    
    # Comment sections you don't want to download
    download_raw_flights()
    download_raw_weather()
    download_processed_data()

    
    print("\n" + "="*50)
    print("✓ Download complete!")
    print("="*50)


if __name__ == "__main__":
    main()