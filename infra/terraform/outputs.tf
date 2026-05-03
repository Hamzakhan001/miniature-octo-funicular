output "ingestion_bucket_name" {
  value = aws_s3_bucket.ingestion.bucket
}

output "ingestion_queue_url" {
  value = aws_sqs_queue.ingestion.id
}

output "ingestion_dlq_url" {
  value = aws_sqs_queue.ingestion_dlq.id
}

output "job_status_table_name" {
  value = aws_dynamodb_table.job_status.name
}

output "lambda_function_name" {
  value = aws_lambda_function.ingestion_router.function_name
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.ingestion.name
}

output "ecs_task_definition_family" {
  value = aws_ecs_task_definition.ingestion_worker.family
}
