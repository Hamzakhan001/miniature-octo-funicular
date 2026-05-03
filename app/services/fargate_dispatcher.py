from __future__ import annotations

import json
from typing import Any

from app.core.config import get_settings
from app.core.logging import logger


class FargateDispatcher:
    def dispatch(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class NoopFargateDispatcher(FargateDispatcher):
    def dispatch(self, payload: dict[str, Any]) -> dict[str, Any]:
        logger.info("fargate_dispatch_skipped", payload=payload)
        return {"backend": "noop", "accepted": True}


class ECSFargateDispatcher(FargateDispatcher):
    def __init__(self) -> None:
        settings = get_settings()
        self._cluster = settings.ecs_cluster
        self._task_definition = settings.ecs_task_definition
        self._container_name = settings.ecs_container_name
        self._subnets = settings.ecs_subnets or []
        self._security_groups = settings.ecs_security_groups or []
        self._assign_public_ip = "ENABLED" if settings.ecs_assign_public_ip else "DISABLED"

        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("boto3 is required for ECS/Fargate dispatch") from exc

        self._client = boto3.client("ecs", region_name=settings.aws_region)

    def dispatch(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.run_task(
            cluster=self._cluster,
            taskDefinition=self._task_definition,
            launchType="FARGATE",
            count=1,
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": self._subnets,
                    "securityGroups": self._security_groups,
                    "assignPublicIp": self._assign_public_ip,
                }
            },
            overrides={
                "containerOverrides": [
                    {
                        "name": self._container_name,
                        "environment": [
                            {"name": "INGESTION_EVENT", "value": json.dumps(payload)},
                        ],
                    }
                ]
            },
        )
        tasks = response.get("tasks", [])
        task_arn = tasks[0]["taskArn"] if tasks else None
        logger.info("fargate_task_dispatched", job_id=payload.get("job_id"), task_arn=task_arn)
        return {"backend": "ecs-fargate", "task_arn": task_arn, "accepted": True}
