#!/usr/bin/env python3
"""
Smart upload: Keep CSV in raw/, auto-convert to Parquet for processed/
Best of both worlds approach
"""

from pathlib import Path
import pandas as pd
import dask.dataframe as dd
from storage import (
    config, 
    upload_file, 
    upload_dataframe,
    upload_directory,
    wait_for_minio
)


def upload_raw_as_csv():
    """Upload raw data in original CSV format (audit trail)"""
    print("\n=== Uploading Raw Data (CSV) ===")
    
    # Keep originals as CSV for reproducibility
    upload_file("data/flight_data_2018_2024.csv", 
                f"{config.raw_flights_path}/flight_data_2018_2024.csv")
    
    upload_directory("data/hourly_for_airport", 
                     config.raw_weather_path)
    
    upload_file("data/airports.csv", 
                f"{config.raw_airports_path}/airports.csv")
    
    print("✓ Raw data preserved as CSV")


def convert_and_upload_parquet():
    """Convert large files to Parquet for analytics (processed layer)"""
    print("\n=== Converting to Parquet for Analytics ===")
    
    # Flight data: Convert to Parquet for fast analytics
    print("\n1. Converting flight data...")
    flight_csv = Path("data/flight_data_2018_2024.csv")
    
    if flight_csv.exists():
        # Define dtypes for problematic columns (mixed types in CSV)
        dtype_spec = {
            'Div1Airport': 'object',
            'Div1TailNum': 'object',
            'Div2Airport': 'object',
            'Div2TailNum': 'object',
            'Div3Airport': 'object',
            'Div3TailNum': 'object',
            'Div4Airport': 'object',
            'Div4TailNum': 'object',
            'Div5Airport': 'object',
            'Div5TailNum': 'object',
            'IATA_Code_Originally_Scheduled_Code_Share_Airline': 'object',
            'Originally_Scheduled_Code_Share_Airline': 'object'
        }
        
        print("   Reading CSV with explicit dtypes...")
        # Use Dask for large files (memory efficient)
        df = dd.read_csv(str(flight_csv), dtype=dtype_spec, assume_missing=True)
        
        print("   Converting to Parquet and uploading...")
        # Save directly to MinIO as Parquet
        parquet_path = config.get_s3_path(f"{config.processed_path}/flight_data.parquet")
        df.to_parquet(
            parquet_path,
            storage_options=config.s3fs_storage_options,
            engine='pyarrow',
            compression='snappy'
        )
        print(f"   ✓ Uploaded: {parquet_path}")
    
    # Weather data: Combine all airport CSVs into single Parquet
    print("\n2. Converting weather data...")
    weather_dir = Path("data/hourly_for_airport")
    
    if weather_dir.exists():
        # Read all weather CSVs with Dask
        weather_pattern = str(weather_dir / "*.csv")
        df_weather = dd.read_csv(weather_pattern)
        
        # Save combined as Parquet
        weather_parquet = config.get_s3_path(f"{config.processed_path}/weather_combined.parquet")
        df_weather.to_parquet(
            weather_parquet,
            storage_options=config.s3fs_storage_options,
            engine='pyarrow',
            compression='snappy'
        )
        print(f"   ✓ Combined {len(list(weather_dir.glob('*.csv')))} files → {weather_parquet}")
    
    print("\n✓ Parquet conversion complete")


def upload_processed_weather_delay():
    """Upload pre-processed weather-delay data"""
    print("\n=== Uploading Processed Weather-Delay Data ===")
    
    weather_delay_dir = Path("data/weather_delay.csv")
    
    if weather_delay_dir.exists() and weather_delay_dir.is_dir():
        # Read Dask partitions and combine to Parquet
        part_files = list(weather_delay_dir.glob("*.part"))
        
        if part_files:
            print(f"Found {len(part_files)} partitions, converting to Parquet...")
            
            # Read all partitions with Dask
            df = dd.read_csv(str(weather_delay_dir / "*.part"))
            
            # Save as single Parquet
            output_path = config.get_s3_path(f"{config.processed_path}/weather_delay_merged.parquet")
            df.to_parquet(
                output_path,
                storage_options=config.s3fs_storage_options,
                engine='pyarrow',
                compression='snappy'
            )
            print(f"   ✓ Merged to: {output_path}")
            return True
    
    print("   ⚠ No processed data found (run analyze.py first)")
    return False


def show_storage_summary():
    """Show what's uploaded and file sizes"""
    from storage import list_files
    
    print("\n" + "="*60)
    print("📊 STORAGE SUMMARY")
    print("="*60)
    
    sections = [
        ("Raw CSV (audit trail)", config.raw_flights_path, config.raw_weather_path),
        ("Processed Parquet (analytics)", config.processed_path, None)
    ]
    
    for name, *paths in sections:
        print(f"\n{name}:")
        for path in paths:
            if path:
                files = list_files(path)
                if files:
                    print(f"  {path}: {len(files)} files")


def compare_formats():
    """Compare CSV vs Parquet file sizes"""
    print("\n=== Format Comparison ===")
    
    # Sample comparison with flight data
    flight_csv = Path("data/flight_data_2018_2024.csv")
    
    if flight_csv.exists():
        csv_size_mb = flight_csv.stat().st_size / (1024**2)
        
        # Read sample to estimate Parquet size
        df_sample = pd.read_csv(flight_csv, nrows=100000)
        df_sample.to_parquet("temp_sample.parquet", compression='snappy')
        parquet_sample_mb = Path("temp_sample.parquet").stat().st_size / (1024**2)
        Path("temp_sample.parquet").unlink()
        
        # Estimate full Parquet size
        total_rows = sum(1 for _ in open(flight_csv)) - 1
        estimated_parquet_mb = (parquet_sample_mb / 100000) * total_rows
        
        print(f"\nFlight Data:")
        print(f"  CSV:              {csv_size_mb:.1f} MB")
        print(f"  Parquet (est):    {estimated_parquet_mb:.1f} MB")
        print(f"  Savings:          {csv_size_mb - estimated_parquet_mb:.1f} MB ({(1 - estimated_parquet_mb/csv_size_mb)*100:.0f}% smaller)")
        print(f"  Speed improvement: ~10-50x faster reads")


def main():
    """
    Upload strategy:
    - Raw: Keep CSV for audit trail
    - Processed: Convert to Parquet for performance
    """
    print("="*60)
    print("📤 SMART UPLOAD: CSV (raw) + Parquet (processed)")
    print("="*60)
    
    wait_for_minio()
    
    # Show size comparison
    compare_formats()
    
    # Ask user preference
    print("\n" + "="*60)
    print("Upload Strategy:")
    print("  1. Hybrid (CSV in raw/, Parquet in processed/) - RECOMMENDED")
    print("  2. Parquet only (save space, faster, no audit trail)")
    print("  3. CSV only (simple, slower, human readable)")
    print("="*60)
    
    choice = input("\nChoice (1/2/3): ").strip()
    
    if choice == "1":
        # Hybrid approach
        upload_raw_as_csv()
        convert_and_upload_parquet()
        upload_processed_weather_delay()
    
    elif choice == "2":
        # Parquet only
        convert_and_upload_parquet()
        upload_processed_weather_delay()
    
    elif choice == "3":
        # CSV only
        upload_raw_as_csv()
        # Also upload processed CSVs
        weather_delay = Path("data/weather_delay.csv")
        if weather_delay.exists():
            upload_directory(str(weather_delay), 
                           f"{config.processed_path}/weather_delay_csv")
    
    else:
        print("Invalid choice, defaulting to hybrid approach")
        upload_raw_as_csv()
        convert_and_upload_parquet()
    
    show_storage_summary()
    
    print("\n" + "="*60)
    print("✅ Upload complete!")
    print(f"View at: http://localhost:9001")
    print("="*60)


if __name__ == "__main__":
    main()