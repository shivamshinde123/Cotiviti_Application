import os
import streamlit as st
from dotenv import load_dotenv

from extractor import extract_text
from guardrail import mask_text
from rag import chunk_text, build_index, retrieve
from llm import answer_question, compare_documents

load_dotenv(override=True)

guardrail_id = os.environ.get("BEDROCK_GUARDRAIL_ID", "")
guardrail_version = os.environ.get("BEDROCK_GUARDRAIL_VERSION", "1")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Policy Insight", layout="wide")
st.title("Policy Insight")
st.caption("Healthcare Content Management Demo")

st.markdown(
    """
    > *Proof of Concept, built for the Cotiviti internship assessment*

    Cotiviti works at the intersection of healthcare data and analytics, helping payers and providers
    make sense of complex clinical and administrative content. This demo explores how generative AI
    and cloud-native services can accelerate that work, safely and at scale.

    **Policy Insight** is a document intelligence tool designed around Cotiviti's core use cases:
    processing medical policies, coding guidelines, and payer contracts while keeping member data private.

    **Capabilities demonstrated:**
    - **PHI / PII Masking:** documents are automatically de-identified using Amazon Bedrock Guardrails
      before any AI model sees the content, reflecting HIPAA-conscious design
    - **Intelligent Q&A:** ask plain-English questions against uploaded documents using a
      RAG pipeline (Amazon Titan Embeddings + FAISS + Amazon Nova Pro)
    - **Version Comparison:** surface what changed between two versions of a policy or contract,
      reducing manual review time for compliance teams

    Built entirely on **AWS Bedrock**, with no third-party model APIs and no data leaving the AWS environment.
    """
)

st.divider()

# ── Sidebarnavigation only ─────────────────────────────────────────────────

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "",
        ["Upload", "Masking Results", "Ask a Question", "Compare Versions"],
        label_visibility="collapsed",
    )

# ── Session state ─────────────────────────────────────────────────────────────

st.session_state.setdefault("docs", [])
st.session_state.setdefault("all_chunks", None)
st.session_state.setdefault("all_index", None)
st.session_state.setdefault("chat_history", [])

# ── Upload page ───────────────────────────────────────────────────────────────

if page == "Upload":
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF or TXT policy documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="doc_uploader",
    )

    if uploaded_files and not guardrail_id:
        st.warning("BEDROCK_GUARDRAIL_ID is not set in your .env file.")

    if uploaded_files and guardrail_id:
        if st.button("Process Documents", type="primary"):
            st.session_state.docs = []
            st.session_state.chat_history = []
            all_masked_text = []

            for f in uploaded_files:
                with st.spinner(f"Processing {f.name}…"):
                    raw = extract_text(f)
                    masked, findings = mask_text(raw, guardrail_id, guardrail_version)
                    st.session_state.docs.append({
                        "name": f.name,
                        "raw": raw,
                        "masked": masked,
                        "findings": findings,
                    })
                    all_masked_text.append(f"--- {f.name} ---\n{masked}")

            combined = "\n\n".join(all_masked_text)
            with st.spinner("Building vector index…"):
                chunks = chunk_text(combined)
                st.session_state.all_chunks = chunks
                st.session_state.all_index = build_index(chunks)

            st.success(f"{len(uploaded_files)} file(s) processed · {len(chunks)} chunks indexed.")

    if st.session_state.docs:
        st.info(f"Currently loaded: {', '.join(d['name'] for d in st.session_state.docs)}")

# ── Masking Results page ──────────────────────────────────────────────────────

elif page == "Masking Results":
    st.header("PHI / PII Masking Results")

    if not st.session_state.docs:
        st.info("Upload and process documents in the Upload page first.")
    else:
        doc_names = [d["name"] for d in st.session_state.docs]
        selected = st.selectbox("Select document to preview", doc_names)
        doc = next(d for d in st.session_state.docs if d["name"] == selected)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Text**")
            st.text_area("", value=doc["raw"][:3000], height=400, disabled=True, key="orig_preview")
        with col2:
            st.markdown("**After Masking**")
            st.text_area("", value=doc["masked"][:3000], height=400, disabled=True, key="masked_preview")

        st.subheader("What Was Masked")
        if doc["findings"]:
            st.dataframe(doc["findings"], use_container_width=True)
        else:
            st.info("No PII/PHI entities detected for this document.")

# ── Ask a Question page ───────────────────────────────────────────────────────

elif page == "Ask a Question":
    st.header("Ask a Question")

    if st.session_state.all_index is None:
        st.info("Upload and process documents in the Upload page first.")
    else:
        if st.session_state.chat_history and st.button("Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("Ask a question about the documents…")
        if question:
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    top_chunks = retrieve(
                        question,
                        st.session_state.all_index,
                        st.session_state.all_chunks,
                    )
                    answer = answer_question(
                        question,
                        top_chunks,
                        st.session_state.chat_history,
                    )
                st.markdown(answer)
                with st.expander("Source chunks used"):
                    for i, chunk in enumerate(top_chunks, 1):
                        st.caption(f"Chunk {i}")
                        st.text(chunk)

            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

# ── Compare Versions page ─────────────────────────────────────────────────────

elif page == "Compare Versions":
    st.header("Compare Versions")

    if len(st.session_state.docs) < 2:
        st.info("Upload at least two documents in the Upload page to compare them.")
    else:
        doc_names = [d["name"] for d in st.session_state.docs]

        col1, col2 = st.columns(2)
        with col1:
            name_a = st.selectbox("Version 1", doc_names, index=0, key="cmp_a")
        with col2:
            name_b = st.selectbox("Version 2", doc_names, index=1, key="cmp_b")

        if name_a == name_b:
            st.warning("Select two different documents to compare.")
        elif st.button("Compare", type="primary"):
            doc_a = next(d for d in st.session_state.docs if d["name"] == name_a)
            doc_b = next(d for d in st.session_state.docs if d["name"] == name_b)

            with st.spinner("Comparing documents…"):
                summary = compare_documents(doc_a["masked"], doc_b["masked"])

            st.subheader("Change Summary")
            st.markdown(summary)

            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{name_a} (masked)**")
                st.text_area("", value=doc_a["masked"][:2000], height=300, disabled=True, key="v1_prev")
            with c2:
                st.markdown(f"**{name_b} (masked)**")
                st.text_area("", value=doc_b["masked"][:2000], height=300, disabled=True, key="v2_prev")
