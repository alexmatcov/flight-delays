import boto3
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
from botocore.client import Config
import pandas as pd


@dataclass
class StorageConfig:
    endpoint_url: str = "http://localhost:9000"
    access_key: str = "admin"
    secret_key: str = "very-intensive-data"
    bucket_name: str = "flight-delays"
    
    raw_flights_path: str = "raw/flights"
    raw_weather_path: str = "raw/weather"
    raw_airports_path: str = "raw/airports"
    processed_path: str = "processed"
    results_path: str = "results"
    
    @property
    def s3fs_storage_options(self) -> dict:
        return {
            "client_kwargs": {"endpoint_url": self.endpoint_url},
            "key": self.access_key,
            "secret": self.secret_key,
        }
    
    def get_s3_path(self, path: str) -> str:
        return f"s3://{self.bucket_name}/{path}"


config = StorageConfig()


def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=config.endpoint_url,
        aws_access_key_id=config.access_key,
        aws_secret_access_key=config.secret_key,
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )


def wait_for_minio(max_retries=30, delay=2):    
    for i in range(max_retries):
        try:
            s3 = get_s3_client()
            s3.list_buckets()
            print("MinIO ready!")
            return s3
        except Exception as e:
            if i < max_retries - 1:
                print(f"  Retry {i+1}/{max_retries}...")
                time.sleep(delay)
            else:
                print(f"\nFailed to connect to MinIO")
                sys.exit(1)


def init_storage():
    s3 = wait_for_minio()
    
    try:
        s3.head_bucket(Bucket=config.bucket_name)
        print(f"Bucket '{config.bucket_name}' exists")
    except:
        s3.create_bucket(Bucket=config.bucket_name)
        print(f"Created bucket '{config.bucket_name}'")
    
    for folder in [config.raw_flights_path, config.raw_weather_path, 
                   config.raw_airports_path, config.processed_path, config.results_path]:
        s3.put_object(Bucket=config.bucket_name, Key=f"{folder}/", Body=b'')
        print(f"✓ {folder}/")
    
    print(f"\nStorage initialized!")
    print(f"Console: http://localhost:9001")
    print(f"  User: {config.access_key} / Pass: {config.secret_key}")


# ============================================================================
# UPLOAD FUNCTIONS
# ============================================================================

def upload_file(local_path: str, s3_key: str, show_progress: bool = True) -> bool:
    """
    Upload a single file to MinIO
    
    Args:
        local_path: Path to local file
        s3_key: Destination path in MinIO (e.g., 'raw/flights/data.csv')
        show_progress: Show upload confirmation
        
    Returns:
        True if successful, False otherwise
    """
    try:
        s3 = get_s3_client()
        s3.upload_file(local_path, config.bucket_name, s3_key)
        if show_progress:
            print(f"✓ Uploaded {local_path} → s3://{config.bucket_name}/{s3_key}")
        return True
    except Exception as e:
        print(f"✗ Failed to upload {local_path}: {e}")
        return False


def upload_dataframe(df: pd.DataFrame, s3_key: str, format: str = 'parquet') -> bool:
    """
    Upload pandas DataFrame directly to MinIO
    
    Args:
        df: Pandas DataFrame
        s3_key: Destination path in MinIO (without extension)
        format: 'parquet' or 'csv'
        
    Returns:
        True if successful
    """
    try:
        s3_path = config.get_s3_path(f"{s3_key}.{format}")
        
        if format == 'parquet':
            df.to_parquet(s3_path, storage_options=config.s3fs_storage_options, 
                         engine='pyarrow', compression='snappy')
        elif format == 'csv':
            df.to_csv(s3_path, storage_options=config.s3fs_storage_options, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        print(f"✓ Uploaded DataFrame → {s3_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to upload DataFrame: {e}")
        return False


def upload_directory(local_dir: str, s3_prefix: str, pattern: str = "*") -> int:
    """
    Upload all files matching pattern from local directory to MinIO
    
    Args:
        local_dir: Local directory path
        s3_prefix: Destination prefix in MinIO (e.g., 'raw/flights')
        pattern: Glob pattern for files (default: all files)
        
    Returns:
        Number of files uploaded
    """
    local_path = Path(local_dir)
    if not local_path.exists():
        print(f"✗ Directory not found: {local_dir}")
        return 0
    
    files = list(local_path.glob(pattern))
    uploaded = 0
    
    print(f"Uploading {len(files)} files from {local_dir}...")
    for file_path in files:
        if file_path.is_file():
            s3_key = f"{s3_prefix}/{file_path.name}"
            if upload_file(str(file_path), s3_key, show_progress=False):
                uploaded += 1
    
    print(f"✓ Uploaded {uploaded}/{len(files)} files to s3://{config.bucket_name}/{s3_prefix}/")
    return uploaded


# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================

def download_file(s3_key: str, local_path: str, show_progress: bool = True) -> bool:
    """
    Download a single file from MinIO
    
    Args:
        s3_key: Source path in MinIO (e.g., 'raw/flights/data.csv')
        local_path: Destination local path
        show_progress: Show download confirmation
        
    Returns:
        True if successful, False otherwise
    """
    try:
        s3 = get_s3_client()
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(config.bucket_name, s3_key, local_path)
        if show_progress:
            print(f"✓ Downloaded s3://{config.bucket_name}/{s3_key} → {local_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {s3_key}: {e}")
        return False


def load_dataframe(s3_key: str, format: str = 'parquet') -> Optional[pd.DataFrame]:
    """
    Load pandas DataFrame directly from MinIO
    
    Args:
        s3_key: Source path in MinIO (with extension)
        format: 'parquet' or 'csv'
        
    Returns:
        DataFrame if successful, None otherwise
    """
    try:
        s3_path = config.get_s3_path(s3_key)
        
        if format == 'parquet':
            df = pd.read_parquet(s3_path, storage_options=config.s3fs_storage_options)
        elif format == 'csv':
            df = pd.read_csv(s3_path, storage_options=config.s3fs_storage_options)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        print(f"✓ Loaded DataFrame from {s3_path} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"✗ Failed to load DataFrame: {e}")
        return None


def download_directory(s3_prefix: str, local_dir: str, pattern: str = "") -> int:
    """
    Download all files from MinIO prefix to local directory
    
    Args:
        s3_prefix: Source prefix in MinIO (e.g., 'raw/flights')
        local_dir: Destination local directory
        pattern: Optional filter for filenames
        
    Returns:
        Number of files downloaded
    """
    try:
        s3 = get_s3_client()
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        
        # List all objects with prefix
        response = s3.list_objects_v2(Bucket=config.bucket_name, Prefix=s3_prefix)
        
        if 'Contents' not in response:
            print(f"No files found at s3://{config.bucket_name}/{s3_prefix}")
            return 0
        
        objects = [obj for obj in response['Contents'] 
                  if not obj['Key'].endswith('/')]  # Skip folders
        
        if pattern:
            objects = [obj for obj in objects if pattern in obj['Key']]
        
        downloaded = 0
        print(f"Downloading {len(objects)} files from {s3_prefix}...")
        
        for obj in objects:
            s3_key = obj['Key']
            filename = Path(s3_key).name
            local_file = local_path / filename
            
            if download_file(s3_key, str(local_file), show_progress=False):
                downloaded += 1
        
        print(f"✓ Downloaded {downloaded}/{len(objects)} files to {local_dir}")
        return downloaded
    except Exception as e:
        print(f"✗ Failed to download directory: {e}")
        return 0


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def list_files(s3_prefix: str = "") -> List[str]:
    """
    List all files in MinIO bucket or specific prefix
    
    Args:
        s3_prefix: Optional prefix to filter (e.g., 'raw/flights')
        
    Returns:
        List of S3 keys
    """
    try:
        s3 = get_s3_client()
        response = s3.list_objects_v2(Bucket=config.bucket_name, Prefix=s3_prefix)
        
        if 'Contents' not in response:
            return []
        
        files = [obj['Key'] for obj in response['Contents'] 
                if not obj['Key'].endswith('/')]
        return files
    except Exception as e:
        print(f"✗ Failed to list files: {e}")
        return []


def file_exists(s3_key: str) -> bool:
    """Check if file exists in MinIO"""
    try:
        s3 = get_s3_client()
        s3.head_object(Bucket=config.bucket_name, Key=s3_key)
        return True
    except:
        return False


def delete_file(s3_key: str) -> bool:
    """Delete file from MinIO"""
    try:
        s3 = get_s3_client()
        s3.delete_object(Bucket=config.bucket_name, Key=s3_key)
        print(f"✓ Deleted s3://{config.bucket_name}/{s3_key}")
        return True
    except Exception as e:
        print(f"✗ Failed to delete {s3_key}: {e}")
        return False


if __name__ == "__main__":
    init_storage()