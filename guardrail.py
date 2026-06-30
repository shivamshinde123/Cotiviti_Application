import os
import streamlit as st
from aws_clients import get_bedrock_runtime

GUARDRAIL_CHUNK_BYTES = 20_000  # Bedrock limit is 25 KB; use 20 KB to stay safe


def validate_guardrail(guardrail_id: str, guardrail_version: str) -> bool:
    """
    Check that the guardrail exists and is reachable before processing a document.
    Returns True if valid, False otherwise (also renders an error in the UI).
    """
    client = get_bedrock_runtime()
    try:
        client.apply_guardrail(
            guardrailIdentifier=guardrail_id,
            guardrailVersion=guardrail_version,
            source="INPUT",
            content=[{"text": {"text": "test"}}],
        )
        return True
    except Exception as e:
        error_msg = str(e)
        region = os.environ.get("AWS_REGION", "us-east-1")

        if "does not exist" in error_msg or "ValidationException" in error_msg:
            st.error(
                f"Guardrail not found. Please check:\n\n"
                f"- **Guardrail ID** — copy the ID from the guardrail detail page (e.g. `9o2awq3821av`)\n"
                f"- **Version** — use `1` if published, or `DRAFT` if not yet published\n"
                f"- **Region** — your `.env` has `AWS_REGION={region}`. "
                f"The guardrail must be in the same region as your app. "
                f"Check the region selector in the AWS Console top-right when viewing the guardrail."
            )
        elif "AccessDeniedException" in error_msg:
            st.error(
                "Access denied. Make sure your IAM user has the `AmazonBedrockFullAccess` policy attached."
            )
        else:
            st.error(f"Guardrail error: {error_msg}")

        return False


def apply_guardrail_to_chunk(client, text: str, guardrail_id: str, guardrail_version: str) -> tuple[str, list[dict]]:
    """Send a single text chunk (under the byte limit) through the guardrail."""
    response = client.apply_guardrail(
        guardrailIdentifier=guardrail_id,
        guardrailVersion=guardrail_version,
        source="INPUT",
        content=[{"text": {"text": text}}],
    )

    outputs = response.get("outputs", [])
    masked = outputs[0].get("text", text) if outputs else text

    findings = []
    for assessment in response.get("assessments", []):
        sensitive = assessment.get("sensitiveInformationPolicy", {})
        for item in sensitive.get("piiEntities", []):
            findings.append({
                "type": item.get("type", "UNKNOWN"),
                "action": item.get("action", ""),
                "match": item.get("match", ""),
            })
        for item in sensitive.get("regexes", []):
            findings.append({
                "type": item.get("name", "REGEX"),
                "action": item.get("action", ""),
                "match": item.get("match", ""),
            })

    return masked, findings


def mask_text(text: str, guardrail_id: str, guardrail_version: str) -> tuple[str, list[dict]]:
    """
    Apply Bedrock Guardrail to anonymize PII/PHI.
    Splits the text into chunks to stay under the 25 KB API limit.
    Returns (masked_text, findings).
    """
    client = get_bedrock_runtime()

    # Split into byte-safe chunks
    encoded = text.encode("utf-8")
    byte_chunks = [
        encoded[i: i + GUARDRAIL_CHUNK_BYTES]
        for i in range(0, len(encoded), GUARDRAIL_CHUNK_BYTES)
    ]
    text_chunks = [c.decode("utf-8", errors="replace") for c in byte_chunks]

    masked_parts = []
    all_findings = []

    try:
        for chunk in text_chunks:
            masked_chunk, findings = apply_guardrail_to_chunk(client, chunk, guardrail_id, guardrail_version)
            masked_parts.append(masked_chunk)
            all_findings.extend(findings)
    except Exception as e:
        st.error(f"Guardrail API error: {e}")
        return text, []

    return "".join(masked_parts), all_findings
