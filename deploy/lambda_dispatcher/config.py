import os


AWS_REGION = os.environ.get("AWS_REGION", "eu-west-2")
ECS_CLUSTER = os.environ.get("ECS_CLUSTER", "")
ECS_TASK_DEFINITION = os.environ.get("ECS_TASK_DEFINITION", "")
ECS_CONTAINER_NAME = os.environ.get("ECS_CONTAINER_NAME", "ingestion-worker")
ECS_SUBNETS = [s.strip() for s in os.environ.get("ECS_SUBNETS", "").split(",") if s.strip()]
ECS_SECURITY_GROUPS = [s.strip() for s in os.environ.get("ECS_SECURITY_GROUPS", "").split(",") if s.strip()]
ECS_ASSIGN_PUBLIC_IP = os.environ.get("ECS_ASSIGN_PUBLIC_IP", "false").lower() == "true"

LAMBDA_MAX_INLINE_FILE_SIZE_MB = int(os.environ.get("LAMBDA_MAX_INLINE_FILE_SIZE_MB", "12"))

JOB_STATUS_BACKEND = os.environ.get("JOB_STATUS_BACKEND", "")
JOB_STATUS_TABLE_NAME = os.environ.get("JOB_STATUS_TABLE_NAME", "")
