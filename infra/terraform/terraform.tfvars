aws_region = "eu-west-2"
project_name = "retrieval-process-docs"
environment = "dev"

vpc_id = "vpc-00f6119ef6111210d"
private_subnet_ids = ["subnet-0d5916ea083464c20", "subnet-0c84481bb5c393cd5"]


container_image = "337480111522.dkr.ecr.eu-west-2.amazonaws.com/ingestion-worker"

openai_api_key_secret_arn = "arn:aws:secretsmanager:eu-west-2:337480111522:secret:openai-api-key-MWqXWR"
vector_store_api_key_secret_arn = "arn:aws:secretsmanager:eu-west-2:337480111522:secret:pinecone-api-key-FTrBHY"


lambda_package_path = "lambda-dispatcher.zip"