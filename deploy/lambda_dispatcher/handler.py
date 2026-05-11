from __future__ import annotations

import json

from router import process_record


def _extract_payloads(record):
    body = json.loads(record["body"])

    # Enriched app-produced message
    if "object_key" in body:
        print(
            json.dumps(
                {
                    "event": "lambda_enriched_payload_received",
                    "job_id": body.get("job_id"),
                    "object_key": body.get("object_key"),
                }
            )
        )
        return [body]

    # Raw S3 -> SQS event
    payloads = []
    for s3_record in body.get("Records", []):
        s3 = s3_record.get("s3", {})
        bucket = s3.get("bucket", {})
        obj = s3.get("object", {})
        key = obj.get("key")
        if not key:
            print(json.dumps({"event": "lambda_s3_record_missing_key"}))
            continue

        parts = key.split("/")
        if len(parts) < 3:
            print(json.dumps({"event": "invalid_s3_object_key", "key": key}))
            continue

        job_id = parts[1]
        filename = parts[-1]
        size = obj.get("size", 0)

        payload = {
            "job_id": job_id,
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

        print(
            json.dumps(
                {
                    "event": "lambda_s3_payload_built",
                    "job_id": job_id,
                    "key": key,
                    "bucket": bucket.get("name"),
                    "filename": filename,
                    "file_size_bytes": size,
                }
            )
        )
        payloads.append(payload)
    return payloads


def handler(event, context):
    failures = []
    processed = 0

    print(
        json.dumps(
            {
                "event": "lambda_event_received",
                "record_count": len(event.get("Records", [])),
            }
        )
    )

    for record in event.get("Records", []):
        message_id = record.get("messageId", "unknown")
        try:
            payloads = _extract_payloads(record)
            print(
                json.dumps(
                    {
                        "event": "lambda_payloads_extracted",
                        "message_id": message_id,
                        "payload_count": len(payloads),
                    }
                )
            )

            for payload in payloads:
                print(
                    json.dumps(
                        {
                            "event": "lambda_processing_payload",
                            "message_id": message_id,
                            "job_id": payload.get("job_id"),
                            "object_key": payload.get("object_key"),
                            "processing_target": payload.get("processing_target"),
                        }
                    )
                )
                process_record(payload)
                processed += 1
                print(
                    json.dumps(
                        {
                            "event": "lambda_payload_processed",
                            "message_id": message_id,
                            "job_id": payload.get("job_id"),
                        }
                    )
                )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "event": "lambda_ingestion_failed",
                        "message_id": message_id,
                        "error": str(exc),
                    }
                )
            )
            failures.append({"itemIdentifier": message_id})

    print(
        json.dumps(
            {
                "event": "lambda_event_completed",
                "processed": processed,
                "failure_count": len(failures),
            }
        )
    )
    return {
        "batchItemFailures": failures,
        "processed": processed,
    }
