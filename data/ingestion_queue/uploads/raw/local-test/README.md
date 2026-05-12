## Overview

This project is a production-oriented Retrieval-Augmented Generation (RAG) application built with FastAPI, Pinecone, OpenAI, and Ragas.

It supports:
- document ingestion
- semantic / hybrid retrieval
- grounded answer generation
- offline benchmark evaluation
- observability with Prometheus and Grafana
- live deployment on AWS EC2 behind Nginx

## Architecture

High-level flow:

1. Documents are ingested and chunked
2. Chunks are embedded with OpenAI embeddings
3. Embeddings are stored in Pinecone
4. User queries go through the RAG pipeline
5. Retrieval returns top-k supporting chunks
6. The generator answers using retrieved context only
7. Metrics, traces, and audit records are captured
8. Offline evaluation is run with Ragas on a benchmark dataset

Core stack:
- FastAPI
- OpenAI
- Pinecone
- Ragas
- Prometheus
- Grafana
- AWS EC2
- Nginx

## Evaluation

This project uses both retrieval and generation evaluation.

### Retrieval metrics
- `hit_rate_at_k`
- `context_recall`
- `avg_docs_retrieved`
- `no_answer_accuracy`

### Generation metrics
- `faithfulness`
- `answer_relevancy`
- `answerable_accuracy`

### Current offline benchmark baseline
- Faithfulness: `0.78`
- Answer Relevancy: `0.62`
- Context Recall: `0.75`

Benchmark evaluation is run offline using a fixed benchmark dataset and Ragas.

## Observability

The app exposes operational monitoring through:
- Prometheus metrics
- Grafana dashboards
- structured audit records
- per-stage latency tracking

Examples of tracked signals:
- query count
- retrieval latency
- generation latency
- docs retrieved
- answer length
- evaluation scores

## Deployment

The application is deployed on AWS EC2 and proxied through Nginx.

Deployment setup:
- FastAPI app running with Uvicorn
- systemd service for process management
- Nginx reverse proxy on port 80
- Pinecone and OpenAI configured through environment variables

## Why this project matters

This project was built to explore what a more production-minded GenAI application looks like beyond a simple demo.

It includes:
- measurable offline evaluation
- retrieval vs generation quality separation
- observability and benchmarking
- deployment to live infrastructure
- source-grounded answers for user trust

## Next Improvements

Planned next steps:
- benchmark threshold gating in CI
- domain + HTTPS
- user feedback collection
- improved source snippet presentation


## Overview

This project is a production-oriented Retrieval-Augmented Generation (RAG) application built with FastAPI, Pinecone, OpenAI, and Ragas.

It supports:
- document ingestion
- semantic / hybrid retrieval
- grounded answer generation
- offline benchmark evaluation
- observability with Prometheus and Grafana
- live deployment on AWS EC2 behind Nginx

## Architecture

High-level flow:

1. Documents are ingested and chunked
2. Chunks are embedded with OpenAI embeddings
3. Embeddings are stored in Pinecone
4. User queries go through the RAG pipeline
5. Retrieval returns top-k supporting chunks
6. The generator answers using retrieved context only
7. Metrics, traces, and audit records are captured
8. Offline evaluation is run with Ragas on a benchmark dataset

Core stack:
- FastAPI
- OpenAI
- Pinecone
- Ragas
- Prometheus
- Grafana
- AWS EC2
- Nginx

## Evaluation

This project uses both retrieval and generation evaluation.

### Retrieval metrics
- `hit_rate_at_k`
- `context_recall`
- `avg_docs_retrieved`
- `no_answer_accuracy`

### Generation metrics
- `faithfulness`
- `answer_relevancy`
- `answerable_accuracy`

### Current offline benchmark baseline
- Faithfulness: `0.78`
- Answer Relevancy: `0.62`
- Context Recall: `0.75`

Benchmark evaluation is run offline using a fixed benchmark dataset and Ragas.

## Observability

The app exposes operational monitoring through:
- Prometheus metrics
- Grafana dashboards
- structured audit records
- per-stage latency tracking

Examples of tracked signals:
- query count
- retrieval latency
- generation latency
- docs retrieved
- answer length
- evaluation scores

## Deployment

The application is deployed on AWS EC2 and proxied through Nginx.

Deployment setup:
- FastAPI app running with Uvicorn
- systemd service for process management
- Nginx reverse proxy on port 80
- Pinecone and OpenAI configured through environment variables

## Why this project matters

This project was built to explore what a more production-minded GenAI application looks like beyond a simple demo.

It includes:
- measurable offline evaluation
- retrieval vs generation quality separation
- observability and benchmarking
- deployment to live infrastructure
- source-grounded answers for user trust

## Next Improvements

Planned next steps:
- benchmark threshold gating in CI
- domain + HTTPS
- user feedback collection
- improved source snippet presentation
