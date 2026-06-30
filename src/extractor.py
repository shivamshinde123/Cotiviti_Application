import io

try:
    import pdfplumber
    _BACKEND = "pdfplumber"
except ImportError:
    import PyPDF2
    _BACKEND = "PyPDF2"


def extract_text(uploaded_file) -> str:
    if uploaded_file.name.lower().endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="replace")

    uploaded_file.seek(0)
    raw = uploaded_file.read()

    if _BACKEND == "pdfplumber":
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    else:
        reader = PyPDF2.PdfReader(io.BytesIO(raw))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
