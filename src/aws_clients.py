import os
import boto3


def get_aws_kwargs() -> dict:
    return {
        "region_name": os.environ.get("AWS_REGION", "us-east-1"),
        "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
    }


def get_bedrock_runtime():
    return boto3.client("bedrock-runtime", **get_aws_kwargs())


def get_bedrock_client():
    return boto3.client("bedrock", **get_aws_kwargs())
