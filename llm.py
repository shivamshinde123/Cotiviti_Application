import json
from aws_clients import get_bedrock_runtime

MODEL_ID = "amazon.nova-pro-v1:0"


def invoke_nova(messages: list[dict], system: str = "") -> str:
    """Call Amazon Nova Pro with a list of {role, content} messages."""
    client = get_bedrock_runtime()

    # Nova expects content as a list of objects, not a plain string
    nova_messages = []
    for msg in messages:
        nova_messages.append({
            "role": msg["role"],
            "content": [{"text": msg["content"]}],
        })

    body: dict = {
        "messages": nova_messages,
        "inferenceConfig": {"max_new_tokens": 1024},
    }
    if system:
        body["system"] = [{"text": system}]

    try:
        resp = client.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        result = json.loads(resp["body"].read())
        return result["output"]["message"]["content"][0]["text"]
    except Exception as e:
        return f"LLM error: {e}"


def answer_question(question: str, context_chunks: list[str], history: list[dict]) -> str:
    """Answer a question using retrieved chunks plus the full conversation history."""
    context = "\n\n---\n\n".join(context_chunks)
    system = (
        "You are a helpful healthcare policy assistant. "
        "Answer questions using only the provided document context. "
        "If the context does not contain the answer, say so clearly."
    )
    grounded_question = (
        f"Use the following document excerpts to answer my question.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}"
    )
    messages = history + [{"role": "user", "content": grounded_question}]
    return invoke_nova(messages, system=system)


def compare_documents(text_a: str, text_b: str) -> str:
    prompt = (
        "Below are two versions of a policy document.\n\n"
        f"VERSION 1:\n{text_a[:4000]}\n\n"
        f"VERSION 2:\n{text_b[:4000]}\n\n"
        "Summarize in plain language what changed between Version 1 and Version 2. "
        "List additions, removals, and modifications using bullet points."
    )
    return invoke_nova(
        [{"role": "user", "content": prompt}],
        system="You are a healthcare compliance analyst.",
    )
