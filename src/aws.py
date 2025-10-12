import boto3
from src.config import Aws

def s3Upload(in_file, out_file):
    s3 = boto3.client('s3')
    s3.upload_file(in_file, Aws.S3_BUCKET_NAME, f"xlsa/{out_file}")
    print(f"Uploaded {in_file} to s3://{Aws.S3_BUCKET_NAME}/xlsa/{out_file}")