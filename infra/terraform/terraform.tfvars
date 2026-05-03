aws_region = "eu-west-2"
project_name = "retrieval-process-docs"
environment = "dev"

vpc_id = "vpc-00f6119ef6111210d"
private_subnet_ids = ["subnet-0944743fc4f05e87b", "subnet-0e305c5a8fe90a8b4"]

container_image = "337480111522.dkr.ecr.eu-west-2.amazonaws.com/ingestion-worker"

openai_api_key_secret_arn = "arn:aws:secretsmanager:eu-west-2:337480111522:secret:openai-api-key-MWqXWR"
vector_store_api_key_secret_arn = "arn:aws:secretsmanager:eu-west-2:337480111522:secret:pinecone-api-key-FTrBHY"
