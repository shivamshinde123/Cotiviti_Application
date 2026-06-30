# Policy Insight: AWS Bedrock Healthcare Demo

> Proof of Concept, built for the Cotiviti internship assessment.

Cotiviti works at the intersection of healthcare data and analytics, helping payers and providers make sense of complex clinical and administrative content. This project explores how generative AI and cloud-native AWS services can accelerate that work, safely and at scale.

**Policy Insight** is a document intelligence tool designed around Cotiviti's core use cases: processing medical policies, coding guidelines, and payer contracts while keeping member data private. It demonstrates a privacy-first RAG pipeline built entirely on AWS Bedrock, with no third-party model APIs and no data leaving the AWS environment.

**Key capabilities:**
- **PHI / PII Masking:** documents are automatically de-identified using Amazon Bedrock Guardrails before any AI model sees the content, reflecting HIPAA-conscious design
- **Intelligent Q&A:** ask plain-English questions against uploaded documents using a RAG pipeline built on Amazon Titan Embeddings, FAISS, and Amazon Nova Pro
- **Version Comparison:** surface what changed between two versions of a policy or contract, reducing manual review time for compliance teams

---

## Features

| Page | What it does |
|---|---|
| **Upload** | Ingest PDF/TXT, redact PHI/PII via Bedrock Guardrails |
| **Masking Results** | View original vs. masked text side-by-side per document |
| **Ask a Question** | RAG Q&A using Titan Embeddings + FAISS + Amazon Nova Pro |
| **Compare Versions** | Select two uploaded docs; Nova Pro summarizes what changed |

---

## Prerequisites

### Step 1: AWS account and IAM access keys

1. Sign in to the [AWS Console](https://console.aws.amazon.com).
2. Click your name (top-right) and select **Security credentials**.
3. Scroll to **Access keys** and click **Create access key**.
4. Select **Local code** as the use case, then click **Next** and **Create access key**.
5. Copy the **Access key ID** and **Secret access key**. The secret is only shown once.

Set `AWS_REGION` to the region where you will use Bedrock. Recommended: `us-east-1` (most model availability).

### Step 2: IAM permissions

Your IAM user must have permission to call Bedrock. The simplest option for a demo:

1. Go to **IAM → Users → your user → Add permissions → Attach policies directly**.
2. Search for and attach **AmazonBedrockFullAccess**.

### Step 3: Enable Bedrock model access

1. Go to **AWS Console → Amazon Bedrock → Model access** (make sure you are in the correct region).
2. Click **Modify model access**.
3. Enable both:
   - **Amazon Titan Embeddings G1 – Text** (`amazon.titan-embed-text-v1`)
   - **Amazon Nova Pro** (`amazon.nova-pro-v1:0`)
4. Submit the request. Approval is usually instant for these models.

### Step 4: Create a Bedrock Guardrail

1. Go to **AWS Console → Amazon Bedrock → Guardrails → Create guardrail**.
2. Give it a name (e.g. `policy-insight-guardrail`).
3. Under **Sensitive information filters**, enable the PII types you want masked: Name, Email, Phone, SSN, Date of Birth, Address (at minimum).
4. Set the action to **Anonymize** for each type.
5. Click through to **Create guardrail** and then **Publish**.
6. Open the guardrail detail page, copy the **Guardrail ID** (e.g. `abc1de2fg3`), and note the **Version** (`1` after publishing, or `DRAFT` before).

---

## Setup

### 1. Fill in your `.env` file

Copy the example and edit it with the values from the steps above:

```bash
cp .env.example .env
```

`.env` fields:

```
AWS_ACCESS_KEY_ID         → from Step 1
AWS_SECRET_ACCESS_KEY     → from Step 1
AWS_REGION                → e.g. us-east-1
BEDROCK_GUARDRAIL_ID      → from Step 4
BEDROCK_GUARDRAIL_VERSION → 1  (or DRAFT)
```

### 2. Install dependencies and run

```bash
uv venv
uv pip install -r requirements.txt
uv run streamlit run app.py
```

---

## Usage

1. Navigate to **Upload** in the sidebar, upload your documents, and click **Process Documents**.
2. Switch to **Masking Results** to review what PHI/PII was detected and redacted per document.
3. Switch to **Ask a Question** to query the documents in a conversational chat interface.
4. Switch to **Compare Versions** to select two uploaded documents and get a plain-English change summary.

---

## Project structure

```
policy-insight/
├── app.py            # Streamlit UI (sidebar navigation, session state)
├── aws_clients.py    # boto3 client factories
├── extractor.py      # PDF / TXT text extraction
├── guardrail.py      # Bedrock Guardrail masking
├── rag.py            # Chunking, Titan embeddings, FAISS
├── llm.py            # Amazon Nova Pro invocation, Q&A, comparison
├── .env              # Your credentials (do not commit)
├── .env.example      # Safe placeholder to commit
├── requirements.txt
└── README.md
```

---

## Notes

- All vectors are stored in memory (FAISS). Refreshing the browser resets the index.
- No authentication or persistent storage. This is a focused proof of concept.
- The LLM context window limits comparison to the first ~4,000 characters of each masked document.
