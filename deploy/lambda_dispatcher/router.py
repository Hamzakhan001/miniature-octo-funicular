import json

import boto3

from config import (
    AWS_REGION,
    ECS_ASSIGN_PUBLIC_IP,
    ECS_CLUSTER,
    ECS_CONTAINER_NAME,
    ECS_SECURITY_GROUPS,
    ECS_SUBNETS,
    ECS_TASK_DEFINITION,
    LAMBDA_MAX_INLINE_FILE_SIZE_MB,
)

ecs = boto3.client("ecs", region_name=AWS_REGION)


def process_record(payload: dict) -> None:
    file_size_bytes = payload.get("file_size_bytes", 0)
    lambda_limit_bytes = LAMBDA_MAX_INLINE_FILE_SIZE_MB * 1024 * 1024

    if file_size_bytes > lambda_limit_bytes:
        dispatch_to_fargate(payload)
        return

    dispatch_to_fargate(payload)


def dispatch_to_fargate(payload: dict) -> None:
    ecs.run_task(
        cluster=ECS_CLUSTER,
        taskDefinition=ECS_TASK_DEFINITION,
        launchType="FARGATE",
        count=1,
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": ECS_SUBNETS,
                "securityGroups": ECS_SECURITY_GROUPS,
                "assignPublicIp": "ENABLED" if ECS_ASSIGN_PUBLIC_IP else "DISABLED",
            }
        },
        overrides={
            "containerOverrides": [
                {
                    "name": ECS_CONTAINER_NAME,
                    "environment": [
                        {
                            "name": "INGESTION_EVENT",
                            "value": json.dumps(payload),
                        }
                    ],
                }
            ]
        },
    )
