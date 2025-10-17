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
        """Options for s3fs/dask to connect to MinIO"""
        return {
            "client_kwargs": {"endpoint_url": self.endpoint_url},
            "key": self.access_key,
            "secret": self.secret_key,
        }
    
    def get_s3_path(self, path: str) -> str:
        """Convert relative path to full s3:// path"""
        return f"s3://{self.bucket_name}/{path}"


config = StorageConfig()


class Storage:
    """Storage client for MinIO operations"""
    
    def __init__(self, config: StorageConfig = config):
        self.config = config
        self.s3 = self._create_client()
    
    def _create_client(self):
        """Create S3 client for MinIO"""
        return boto3.client(
            's3',
            endpoint_url=self.config.endpoint_url,
            aws_access_key_id=self.config.access_key,
            aws_secret_access_key=self.config.secret_key,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )
    
    def wait_for_minio(self, max_retries=30, delay=2):
        """Wait for MinIO to be ready before proceeding"""
        for i in range(max_retries):
            try:
                self.s3.list_buckets()
                print("MinIO ready!")
                return True
            except Exception as e:
                if i < max_retries - 1:
                    print(f"  Retry {i+1}/{max_retries}...")
                    time.sleep(delay)
                else:
                    print("\nFailed to connect to MinIO")
                    sys.exit(1)
        return False
    
    def init_storage(self):
        """Initialize MinIO bucket and folder structure"""
        self.wait_for_minio()
        
        try:
            self.s3.head_bucket(Bucket=self.config.bucket_name)
            print(f"Bucket '{self.config.bucket_name}' exists")
        except:
            self.s3.create_bucket(Bucket=self.config.bucket_name)
            print(f"Created bucket '{self.config.bucket_name}'")
        
        for folder in [self.config.raw_flights_path, self.config.raw_weather_path, 
                       self.config.raw_airports_path, self.config.processed_path, 
                       self.config.results_path]:
            self.s3.put_object(Bucket=self.config.bucket_name, Key=f"{folder}/", Body=b'')
            print(f"✓ {folder}/")
        
        print("\nStorage initialized!")
        print("Console: http://localhost:9001")
        print(f"  User: {self.config.access_key} / Pass: {self.config.secret_key}")
    
    def upload_file(self, local_path: str, s3_key: str, show_progress: bool = True) -> bool:
        """Upload a single file to MinIO"""
        try:
            self.s3.upload_file(local_path, self.config.bucket_name, s3_key)
            if show_progress:
                print(f"✓ Uploaded {local_path} → s3://{self.config.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            print(f"✗ Failed to upload {local_path}: {e}")
            return False
    
    def upload_dataframe(self, df: pd.DataFrame, s3_key: str, format: str = 'parquet') -> bool:
        """Upload pandas DataFrame directly to MinIO"""
        try:
            s3_path = self.config.get_s3_path(f"{s3_key}.{format}")
            
            if format == 'parquet':
                df.to_parquet(s3_path, storage_options=self.config.s3fs_storage_options, 
                             engine='pyarrow', compression='snappy')
            elif format == 'csv':
                df.to_csv(s3_path, storage_options=self.config.s3fs_storage_options, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            print(f"✓ Uploaded DataFrame → {s3_path}")
            return True
        except Exception as e:
            print(f"✗ Failed to upload DataFrame: {e}")
            return False
    
    def upload_directory(self, local_dir: str, s3_prefix: str, pattern: str = "*") -> int:
        """Upload all files matching pattern from local directory to MinIO"""
        local_path = Path(local_dir)
        if not local_path.exists():
            print(f"✗ Directory not found: {local_dir}")
            return 0
        
        files = [f for f in local_path.glob(pattern) if f.is_file()]
        if not files:
            print(f"✗ No files found matching pattern: {pattern}")
            return 0
        
        print(f"Uploading {len(files)} files from {local_dir}...")
        uploaded = 0
        for file_path in files:
            s3_key = f"{s3_prefix}/{file_path.name}"
            if self.upload_file(str(file_path), s3_key, show_progress=False):
                uploaded += 1
        
        print(f"✓ Uploaded {uploaded}/{len(files)} files to s3://{self.config.bucket_name}/{s3_prefix}/")
        return uploaded
    
    def download_file(self, s3_key: str, local_path: str, show_progress: bool = True) -> bool:
        """Download a single file from MinIO"""
        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(self.config.bucket_name, s3_key, local_path)
            if show_progress:
                print(f"✓ Downloaded s3://{self.config.bucket_name}/{s3_key} → {local_path}")
            return True
        except Exception as e:
            print(f"✗ Failed to download {s3_key}: {e}")
            return False
    
    def load_dataframe(self, s3_key: str, format: str = 'parquet') -> Optional[pd.DataFrame]:
        """Load pandas DataFrame directly from MinIO"""
        try:
            s3_path = self.config.get_s3_path(s3_key)
            
            if format == 'parquet':
                df = pd.read_parquet(s3_path, storage_options=self.config.s3fs_storage_options)
            elif format == 'csv':
                df = pd.read_csv(s3_path, storage_options=self.config.s3fs_storage_options)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            print(f"✓ Loaded DataFrame from {s3_path} ({len(df)} rows)")
            return df
        except Exception as e:
            print(f"✗ Failed to load DataFrame: {e}")
            return None
    
    def download_directory(self, s3_prefix: str, local_dir: str, pattern: str = "") -> int:
        """Download all files from MinIO prefix to local directory"""
        try:
            local_path = Path(local_dir)
            local_path.mkdir(parents=True, exist_ok=True)
            
            response = self.s3.list_objects_v2(Bucket=self.config.bucket_name, Prefix=s3_prefix)
            
            if 'Contents' not in response:
                print(f"No files found at s3://{self.config.bucket_name}/{s3_prefix}")
                return 0
            
            objects = [obj for obj in response['Contents'] 
                      if not obj['Key'].endswith('/')]
            
            if pattern:
                objects = [obj for obj in objects if pattern in obj['Key']]
            
            if not objects:
                print(f"No files found matching pattern: {pattern}")
                return 0
            
            print(f"Downloading {len(objects)} files from {s3_prefix}...")
            downloaded = 0
            
            for obj in objects:
                s3_key = obj['Key']
                filename = Path(s3_key).name
                local_file = local_path / filename
                
                if self.download_file(s3_key, str(local_file), show_progress=False):
                    downloaded += 1
            
            print(f"✓ Downloaded {downloaded}/{len(objects)} files to {local_dir}")
            return downloaded
        except Exception as e:
            print(f"✗ Failed to download directory: {e}")
            return 0
    
    def list_files(self, s3_prefix: str = "") -> List[str]:
        """List all files in MinIO bucket or specific prefix"""
        try:
            response = self.s3.list_objects_v2(Bucket=self.config.bucket_name, Prefix=s3_prefix)
            
            if 'Contents' not in response:
                return []
            
            files = [obj['Key'] for obj in response['Contents'] 
                    if not obj['Key'].endswith('/')]
            return files
        except Exception as e:
            print(f"✗ Failed to list files: {e}")
            return []
    
    def file_exists(self, s3_key: str) -> bool:
        """Check if file exists in MinIO"""
        try:
            self.s3.head_object(Bucket=self.config.bucket_name, Key=s3_key)
            return True
        except:
            return False
    
    def delete_file(self, s3_key: str) -> bool:
        """Delete file from MinIO"""
        try:
            self.s3.delete_object(Bucket=self.config.bucket_name, Key=s3_key)
            print(f"✓ Deleted s3://{self.config.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            print(f"✗ Failed to delete {s3_key}: {e}")
            return False


if __name__ == "__main__":
    storage = Storage()
    storage.init_storage()