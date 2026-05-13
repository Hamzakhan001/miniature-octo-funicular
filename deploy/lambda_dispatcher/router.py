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
    JOB_STATUS_BACKEND,
    JOB_STATUS_TABLE_NAME,
    LAMBDA_MAX_INLINE_FILE_SIZE_MB,
)

ecs = boto3.client("ecs", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION) if JOB_STATUS_BACKEND == "dynamodb" and JOB_STATUS_TABLE_NAME else None
job_table = dynamodb.Table(JOB_STATUS_TABLE_NAME) if dynamodb else None


def process_record(payload: dict) -> None:
    job_id = payload["job_id"]
    file_size_bytes = payload.get("file_size_bytes", 0)
    lambda_limit_bytes = LAMBDA_MAX_INLINE_FILE_SIZE_MB * 1024 * 1024

    update_job_status(job_id, "queued")
    mark_stage_raw(job_id, "lambda_received_at", "lambda_received")
    if file_size_bytes > lambda_limit_bytes:
        update_job_status(job_id, "processing_fargate")
        dispatch_to_fargate(payload)
        return

    update_job_status(job_id, "processing_fargate")
    mark_stage_raw(job_id, "fargate_dispatched_at", "fargate_dispatched")
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


def update_job_status(job_id: str, status: str) -> None:
    if not job_table:
        return

    job_table.update_item(
        Key={"job_id": job_id},
        UpdateExpression="SET #status = :status, #updated_at = :updated_at",
        ExpressionAttributeNames={
            "#status": "status",
            "#updated_at": "updated_at",
        },
        ExpressionAttributeValues={
            ":status": status,
            ":updated_at": _utc_now(),
        },
    )

def mark_stage_raw(job_id: str, stage: str, current_stage: str) -> None:
    if not job_table:
        return 
    
    now = _utc_now()
    job_table.update_item(
            Key = {"job_id": job_id},
        UpdateExpression=(
            "SET #stage_timestamps.#stage = :ts, "
            "#progress.#current_stage = :current_stage, "
            "#updated_at = :updated_at"
        ),
        ExpressionAttributeNames={
            "#stage_timestamps": "stage_timestamps",
            "#stage": stage,
            "#progress": "progress",
            "#current_stage": current_stage,
            "#updated_at": "updated_at",
        },
        ExpressionAttributeValues={
            ":ts": now,
            ":current_stage": current_stage,
            ":updated_at": now,
        },
    )


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
