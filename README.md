# Production RAG Pipeline

A production-grade Retrieval-Augmented Generation system with event-driven async document ingestion, vector search, automated evaluation, and full observability. Built for real workloads — not a tutorial demo.

**Live demo:** [rag.hamzatwin.site](http://rag.hamzatwin.site)

---

## What this is

Most RAG projects are a Jupyter notebook with `retriever.get_relevant_documents()`. This is the full engineering picture around that — async ingestion, cloud infrastructure, evaluation in production, and metrics you can actually act on.

---

## Architecture

```
                         ┌──────────────┐
                         │   Next.js    │
                         │   Frontend   │
                         └──────┬───────┘
                                │
                         ┌──────▼───────┐
                         │  FastAPI     │
                         │  Backend     │
                         └──────┬───────┘
                                │
                    ┌───────────▼───────────┐
                    │     Amazon S3         │
                    │  (document storage)   │
                    └───────────┬───────────┘
                                │ ObjectCreated event
                    ┌───────────▼───────────┐
                    │     Amazon SQS        │
                    │   (ingestion queue)   │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   AWS Lambda Router   │
                    │  (size-based routing) │
                    └─────┬─────────┬───────┘
                          │         │
              small files │         │ large files (>1MB)
                          │         │
               ┌──────────▼─┐   ┌───▼──────────────┐
               │   Lambda   │   │   ECS Fargate     │
               │ (inline)   │   │ (containerised)   │
               └──────────┬─┘   └───┬───────────────┘
                          │         │
                          └────┬────┘
                               │
                   ┌───────────▼───────────┐
                   │  Chunk → Embed →      │
                   │  Upsert (bounded      │
                   │  concurrency)         │
                   └───────────┬───────────┘
                               │
                   ┌───────────▼───────────┐
                   │  Pinecone Vector DB   │
                   └───────────────────────┘
```

**Query path:**
```
User → Input Guardrail → Vector Retrieve → Rerank → LLM → Output Guardrail → RAGAS Eval → Audit Log → Response
```

---

## Key engineering decisions

**Dual-path routing (Lambda vs Fargate)**
A 23MB PDF produces ~32,000 chunks and takes ~19 minutes to embed and upsert. Lambda's 15-minute timeout makes this impossible inline. The router dispatches large files to Fargate and processes small files directly in Lambda — keeping fast ingestion fast and not blocking large jobs.

**SQS decoupling**
The queue absorbs upload bursts, gives automatic retries with visibility timeout, and isolates failures to a DLQ. The API returns immediately after queuing — users aren't waiting on embedding calls.

**Bounded concurrency on Pinecone upserts**
Pinecone rate-limits write throughput. A semaphore caps concurrent batch upserts, avoiding 429 cascades across hundreds of batches without exponential backoff failures.

**chunk_size=384**
Smaller chunks preserve clause-level granularity. Each chunk maps to one coherent idea, improving faithfulness scores in RAGAS evaluation. Larger windows bundle unrelated content into one vector, degrading retrieval precision.

**Pushgateway for Fargate metrics**
Fargate tasks are ephemeral — they exit after processing. Prometheus cannot scrape a dead container. Metrics are pushed to a Pushgateway (behind a Network Load Balancer) at job completion, making ingestion throughput and latency visible in Grafana without parsing logs.

**RAGAS evaluation on every query**
Faithfulness, answer relevance, and context coverage are scored per query in production. Scores are emitted as Prometheus histograms, making quality degradation visible before users report it.

**Audit trail**
Every query logs input, retrieved chunks, prompt sent to LLM, response, guardrail decision, and eval scores. Required for regulated industry use cases where AI decisions must be explainable.

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI, Python 3.13 |
| Frontend | Next.js 14, TypeScript |
| Queue | AWS SQS + DLQ |
| Ingestion worker | AWS ECS Fargate |
| Routing | AWS Lambda |
| Storage | AWS S3 |
| Job state | AWS DynamoDB |
| Vector store | Pinecone |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| Evaluation | RAGAS (faithfulness, relevance, context coverage) |
| Metrics | Prometheus + Grafana + Pushgateway (NLB) |
| Tracing | OpenTelemetry |
| Secrets | AWS Secrets Manager |
| Infrastructure | Terraform |
| Container registry | AWS ECR |

---

## Real production numbers

| Document | Size | Chunks | Batches | Latency |
|---|---|---|---|---|
| UK visa register PDF | 23MB | 32,491 | 650 | ~19 min |
| Witness statement | 100KB | 16 | 1 | 4s |
| NDA | 34KB | 11–13 | 1 | ~3s |

Cumulative across all runs: **260,032 vectors** in Pinecone, **5,208 upsert batches**, **16 ingestion jobs tracked in DynamoDB**.

---

## RAGAS eval scores (production query)

Query against the ingested corpus:

```
faithfulness:       0.692
answer_relevance:   0.800
context_coverage:   0.655
latency:            4,134ms
```

Scores are tracked per query in Prometheus and visualised in Grafana.

---

## Project structure

```
app/
├── api/
│   ├── routes/          # ingest, query, evaluation, audit, health
│   └── middleware.py    # auth, request ID, request logging
├── rag/
│   ├── pipeline.py      # end-to-end query pipeline with observability
│   ├── ingestion.py     # chunking, embedding, batched upserts
│   ├── vectorstore.py   # Pinecone client
│   └── reranker.py      # result reranking
├── services/
│   ├── event_processor.py       # SQS event handler
│   ├── fargate_dispatcher.py    # ECS task launcher
│   ├── job_repository.py        # DynamoDB job state
│   └── queue_backend.py         # SQS producer
├── observability/
│   ├── metrics.py       # Prometheus counters and histograms
│   ├── audit.py         # query audit records
│   └── tracing.py       # OpenTelemetry spans
├── guardrails/
│   ├── input_guard.py   # query validation before retrieval
│   └── output_guard.py  # response validation before return
├── evals/
│   └── runner.py        # RAGAS evaluation runner
└── workers/
    ├── fargate_entrypoint.py   # Fargate task entry point
    └── lambda_handler.py       # Lambda SQS handler

infra/terraform/           # Full AWS infrastructure as code
deploy/lambda_dispatcher/  # Lambda routing logic
```

---

## Security

- API keys stored in AWS Secrets Manager, never in environment variables directly
- Lambda and ECS use least-privilege IAM roles scoped to specific resources
- Queue messages carry metadata only — document bytes stay in S3
- Fargate runs in private subnets with NAT for outbound calls
- Per-client data isolation via Pinecone namespaces

---

## Running locally

```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env  # add OpenAI, Pinecone, AWS credentials

# Start API
uvicorn app.main:app --reload --port 8000

# Start observability stack (Prometheus + Grafana + Pushgateway)
docker compose up -d
```

API docs: `http://localhost:8000/docs`
Grafana: `http://localhost:3001` (admin / admin)

---

## Infrastructure

All AWS resources are defined in Terraform:

```bash
cd infra/terraform
terraform init
terraform plan
terraform apply
```

Provisions: S3, SQS + DLQ, DynamoDB, Lambda, ECS cluster + task definition, ECR, Pushgateway NLB, IAM roles, Secrets Manager references.
