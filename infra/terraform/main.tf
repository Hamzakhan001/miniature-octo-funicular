locals {
  name_prefix = "${var.project_name}-${var.environment}"
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

resource "aws_s3_bucket" "ingestion" {
  bucket = "${local.name_prefix}-ingestion"
  tags   = local.common_tags
}

resource "aws_s3_bucket_versioning" "ingestion" {
  bucket = aws_s3_bucket.ingestion.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "ingestion" {
  bucket = aws_s3_bucket.ingestion.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "ingestion" {
  bucket = aws_s3_bucket.ingestion.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_cors_configuration" "ingestion" {
  bucket = aws_s3_bucket.ingestion.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["PUT", "POST", "GET", "HEAD"]
    allowed_origins = ["*"]
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }
}

resource "aws_sqs_queue" "ingestion_dlq" {
  name = "${local.name_prefix}-ingestion-dlq"
  tags = local.common_tags
}

resource "aws_sqs_queue" "ingestion" {
  name                       = "${local.name_prefix}-ingestion"
  visibility_timeout_seconds = 330
  message_retention_seconds  = 345600
  receive_wait_time_seconds  = 20

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.ingestion_dlq.arn
    maxReceiveCount     = 5
  })

  tags = local.common_tags
}

resource "aws_sqs_queue_policy" "allow_s3" {
  queue_url = aws_sqs_queue.ingestion.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowS3ToSendMessages"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
        Action   = "sqs:SendMessage"
        Resource = aws_sqs_queue.ingestion.arn
        Condition = {
          ArnEquals = {
            "aws:SourceArn" = aws_s3_bucket.ingestion.arn
          }
        }
      }
    ]
  })
}

resource "aws_s3_bucket_notification" "ingestion" {
  bucket = aws_s3_bucket.ingestion.id

  queue {
    queue_arn = aws_sqs_queue.ingestion.arn
    events    = ["s3:ObjectCreated:*"]
  }

  depends_on = [aws_sqs_queue_policy.allow_s3]
}

resource "aws_dynamodb_table" "job_status" {
  name         = "${local.name_prefix}-ingestion-jobs"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "job_id"

  attribute {
    name = "job_id"
    type = "S"
  }

  tags = local.common_tags
}

resource "aws_iam_role" "lambda_role" {
  name = "${local.name_prefix}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_security_group" "fargate_tasks" {
  name        = "${local.name_prefix}-fargate-sg"
  description = "Security group for ingestion Fargate tasks"
  vpc_id      = var.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

resource "aws_ecs_cluster" "ingestion" {
  name = "${local.name_prefix}-ecs-cluster"
  tags = local.common_tags
}

resource "aws_cloudwatch_log_group" "lambda_logs" {
  name              = "/aws/lambda/${local.name_prefix}-ingestion-router"
  retention_in_days = 14
  tags              = local.common_tags
}

resource "aws_cloudwatch_log_group" "ecs_logs" {
  name              = "/ecs/${local.name_prefix}-ingestion-worker"
  retention_in_days = 14
  tags              = local.common_tags
}

resource "aws_iam_role" "ecs_task_execution_role" {
  name = "${local.name_prefix}-ecs-task-exec-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_basic" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_task_execution_secrets" {
  name = "${local.name_prefix}-ecs-task-exec-secrets"
  role = aws_iam_role.ecs_task_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["secretsmanager:GetSecretValue"]
        Resource = [
          var.openai_api_key_secret_arn,
          var.vector_store_api_key_secret_arn
        ]
      }
    ]
  })
}

resource "aws_iam_role" "ecs_task_role" {
  name = "${local.name_prefix}-ecs-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "ecs_task_policy" {
  name = "${local.name_prefix}-ecs-task-policy"
  role = aws_iam_role.ecs_task_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
        {
            Effect = "Allow"
            Action = [
            "s3:GetObject",
            "s3:GetObjectAttributes",
            "s3:HeadObject"
            ]
            Resource = "${aws_s3_bucket.ingestion.arn}/*"
        },
        {
            Effect = "Allow"
            Action = [
            "secretsmanager:GetSecretValue"
            ]
            Resource = [
            var.openai_api_key_secret_arn,
            var.vector_store_api_key_secret_arn
            ]
        },
        {
            Effect = "Allow"
            Action = [
            "dynamodb:GetItem",
            "dynamodb:PutItem",
            "dynamodb:UpdateItem"
            ]
            Resource = aws_dynamodb_table.job_status.arn
        }
        ]

  })
}

resource "aws_ecs_task_definition" "ingestion_worker" {
  family                   = "${local.name_prefix}-ingestion-worker"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = tostring(var.fargate_cpu)
  memory                   = tostring(var.fargate_memory)
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name      = "ingestion-worker"
      image     = var.container_image
      essential = true
      command   = ["python", "-m", "app.workers.fargate_entrypoint"]
      environment = [
        { name = "STORAGE_BACKEND", value = "s3" },
        { name = "AWS_REGION", value = var.aws_region },
        { name = "S3_INGESTION_BUCKET", value = aws_s3_bucket.ingestion.bucket },
        { name = "JOB_STATUS_BACKEND", value = "dynamodb" },
        { name = "JOB_STATUS_TABLE_NAME", value = aws_dynamodb_table.job_status.name },
        { name = "OPENAI_API_KEY_SECRET_ARN", value = var.openai_api_key_secret_arn },
        { name = "VECTOR_STORE_API_KEY_SECRET_ARN", value = var.vector_store_api_key_secret_arn }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.ecs_logs.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])

  tags = local.common_tags
}

resource "aws_iam_policy" "lambda_ingestion_policy" {
  name = "${local.name_prefix}-lambda-ingestion-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectAttributes",
          "s3:HeadObject"
        ]
        Resource = "${aws_s3_bucket.ingestion.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes",
          "sqs:ChangeMessageVisibility"
        ]
        Resource = aws_sqs_queue.ingestion.arn
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem"
        ]
        Resource = aws_dynamodb_table.job_status.arn
      },
      {
        Effect = "Allow"
        Action = ["secretsmanager:GetSecretValue"]
        Resource = [
          var.openai_api_key_secret_arn,
          var.vector_store_api_key_secret_arn
        ]
      },
      {
        Effect = "Allow"
        Action = ["ecs:RunTask"]
        Resource = aws_ecs_task_definition.ingestion_worker.arn
      },
      {
        Effect = "Allow"
        Action = ["iam:PassRole"]
        Resource = [
          aws_iam_role.ecs_task_execution_role.arn,
          aws_iam_role.ecs_task_role.arn
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_ingestion_attach" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = aws_iam_policy.lambda_ingestion_policy.arn
}

resource "aws_lambda_function" "ingestion_router" {
  function_name = "${local.name_prefix}-ingestion-router"
  role          = aws_iam_role.lambda_role.arn
  handler       = "handler.handler"
  runtime       = "python3.13"
  timeout       = 300
  memory_size   = 1024

  filename         = var.lambda_package_path
  source_code_hash = filebase64sha256(var.lambda_package_path)

  environment {
    variables = {
      STORAGE_BACKEND                 = "s3"
      QUEUE_BACKEND                   = "sqs"
      S3_INGESTION_BUCKET             = aws_s3_bucket.ingestion.bucket
      SQS_INGESTION_QUEUE_URL         = aws_sqs_queue.ingestion.id
      ECS_CLUSTER                     = aws_ecs_cluster.ingestion.name
      ECS_TASK_DEFINITION             = aws_ecs_task_definition.ingestion_worker.family
      ECS_CONTAINER_NAME              = "ingestion-worker"
      ECS_SUBNETS                     = join(",", var.private_subnet_ids)
      ECS_SECURITY_GROUPS             = aws_security_group.fargate_tasks.id
      ECS_ASSIGN_PUBLIC_IP            = "false"
      LAMBDA_MAX_INLINE_FILE_SIZE_MB  = tostring(var.lambda_max_inline_file_size_mb)
      INGESTION_MAX_FILE_SIZE_MB      = tostring(var.ingestion_max_file_size_mb)
      LAMBDA_SUPPORTED_EXTENSIONS     = join(",", var.lambda_supported_extensions)
      FARGATE_PREFERRED_EXTENSIONS    = join(",", var.fargate_preferred_extensions)
      JOB_STATUS_TABLE_NAME           = aws_dynamodb_table.job_status.name
      OPENAI_API_KEY_SECRET_ARN       = var.openai_api_key_secret_arn
      VECTOR_STORE_API_KEY_SECRET_ARN = var.vector_store_api_key_secret_arn
    }
  }

  tags = local.common_tags
}

resource "aws_lambda_event_source_mapping" "ingestion" {
  event_source_arn        = aws_sqs_queue.ingestion.arn
  function_name           = aws_lambda_function.ingestion_router.arn
  batch_size              = 5
  function_response_types = ["ReportBatchItemFailures"]
}
