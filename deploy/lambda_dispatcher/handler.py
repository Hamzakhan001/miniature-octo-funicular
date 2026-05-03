import json

from router import process_record


def handler(event, context):
    failures = []
    processed = 0

    for record in event.get("Records", []):
        message_id = record.get("messageId", "unknown")
        try:
            body = json.loads(record["body"])
            process_record(body)
            processed += 1
        except Exception:
            failures.append({"itemIdentifier": message_id})

    return {
        "batchItemFailures": failures,
        "processed": processed,
    }
