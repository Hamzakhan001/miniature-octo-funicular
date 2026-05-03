variable "aws_region" {
    type = string
    default = "eu-west-2"
}

variable "project_name" {
    type = string
    default = "retrieval-process-docs"
}

variable "environment" {
    type = string
    default = "dev"
}

variable "vpc_id" {
  type = string
}

variable "private_subnet_ids" {
  type = list(string)
}

variable "container_image" {
  type = string
}

variable "openai_api_key_secret_arn" {
  type = string
}

variable "vector_store_api_key_secret_arn" {
  type = string
}

variable "lambda_package_path" {
  type    = string
  default = "lambda.zip"
}

variable "lambda_max_inline_file_size_mb" {
  type    = number
  default = 12
}

variable "ingestion_max_file_size_mb" {
  type    = number
  default = 100
}

variable "lambda_supported_extensions" {
  type    = list(string)
  default = [".txt", ".md", ".html", ".json"]
}

variable "fargate_preferred_extensions" {
  type    = list(string)
  default = [".pdf", ".docx", ".csv"]
}

variable "fargate_cpu" {
  type    = number
  default = 1024
}

variable "fargate_memory" {
  type    = number
  default = 2048
}