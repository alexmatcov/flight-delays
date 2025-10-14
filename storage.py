import boto3
import time
import sys
from dataclasses import dataclass
from botocore.client import Config
from botocore.exceptions import ClientError, EndpointConnectionError


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


def wait_for_minio(max_retries=30, delay=2):    
    for i in range(max_retries):
        try:
            s3 = boto3.client(
                's3',
                endpoint_url=config.endpoint_url,
                aws_access_key_id=config.access_key,
                aws_secret_access_key=config.secret_key,
                config=Config(signature_version='s3v4'),
                region_name='us-east-1'
            )
            s3.list_buckets()
            print("MinIO ready!")
            return s3
        except Exception as e:
            if i < max_retries - 1:
                print(f"  Retry {i+1}/{max_retries}...")
                time.sleep(delay)
            else:
                print(f"\nFailed to connect")
                sys.exit(1)


def init_storage():
    s3 = wait_for_minio()
    
    try:
        s3.head_bucket(Bucket=config.bucket_name)
        print(f"Bucket exists")
    except:
        s3.create_bucket(Bucket=config.bucket_name)
        print(f"Created bucket")
    
    for folder in [config.raw_flights_path, config.raw_weather_path, 
                   config.raw_airports_path, config.processed_path, config.results_path]:
        s3.put_object(Bucket=config.bucket_name, Key=f"{folder}/", Body=b'')
        print(f"✓ {folder}/")
    
    print(f"\nDone! Console: http://localhost:9001")
    print(f"  User: {config.access_key} / Pass: {config.secret_key}")


if __name__ == "__main__":
    init_storage()