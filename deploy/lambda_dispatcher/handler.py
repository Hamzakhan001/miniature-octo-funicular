import json

from router import process_record


def _extract_payloads(record):
    body = json.loads(record["body"])

    # Enriched app-produced message
    if "object_key" in body:
        return [body]

    # Raw S3 -> SQS event
    payloads = []
    for s3_record in body.get("Records", []):
        s3 = s3_record.get("s3", {})
        bucket = s3.get("bucket", {})
        obj = s3.get("object", {})
        key = obj.get("key")
        if not key:
            continue

        filename = key.split("/")[-1]
        size = obj.get("size", 0)

        payloads.append(
            {
                "job_id": f"s3::{bucket.get('name', 'unknown')}::{key}",
                "filename": filename,
                "object_key": key,
                "file_size_bytes": size,
                "processing_target": "fargate",
                "metadata": {
                    "bucket": bucket.get("name"),
                    "event_name": s3_record.get("eventName"),
                    "event_time": s3_record.get("eventTime"),
                    "source": "s3_event",
                },
            }
        )
    return payloads


def handler(event, context):
    failures = []
    processed = 0

    for record in event.get("Records", []):
        message_id = record.get("messageId", "unknown")
        try:
            payloads = _extract_payloads(record)
            for payload in payloads:
                process_record(payload)
                processed += 1
        except Exception:
            failures.append({"itemIdentifier": message_id})

    return {
        "batchItemFailures": failures,
        "processed": processed,
    }
