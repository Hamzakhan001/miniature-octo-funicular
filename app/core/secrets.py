from __future__ import annotations

import json

import boto3

from app.core.config import get_settings


def get_secret_value(secret_arn: str) -> str:
    settings = get_settings()
    client = boto3.client("secretsmanager", region_name=settings.aws_region)
    response = client.get_secret_value(SecretId=secret_arn)
    secret_string = response.get("SecretString", "")
    if not secret_string:
        raise ValueError(f"Secret {secret_arn} has no SecretString")

    try:
        parsed = json.loads(secret_string)
    except json.JSONDecodeError:
        return secret_string

    if "OPENAI_API_KEY" in parsed:
        return parsed["OPENAI_API_KEY"]
    if "PINECONE_API_KEY" in parsed:
        return parsed["PINECONE_API_KEY"]

    if len(parsed) == 1:
        return next(iter(parsed.values()))

    raise ValueError(f"Could not determine secret value for {secret_arn}")
