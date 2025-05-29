import streamlit as st
from transformers import pipeline
import fitz  # PyMuPDF

# Load the summarizer (cached for performance)
@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")
str
summarizer = load_model()

st.title("📄 AI PDF/Text Summarizer")
st.write("Upload a PDF or paste text to generate a summary.")

# PDF or text input
pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
text_input = st.text_area("Or paste text here", height=300)

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Summarize
if st.button("Summarize"):
    if pdf_file:
        with st.spinner("Extracting and summarizing PDF..."):
            raw_text = extract_text_from_pdf(pdf_file)
    elif text_input.strip():
        raw_text = text_input
    else:
        st.warning("Please upload a PDF or enter some text.")
        st.stop()

    # Split if too long
    if len(raw_text) > 1000:
        import textwrap
        chunks = textwrap.wrap(raw_text, 1000)
        summary = []
        for chunk in chunks:
            out = summarizer(chunk, max_length=60, min_length=20, do_sample=False)
            summary.append(out[0]['summary_text'])
        final_summary = " ".join(summary)
    else:
        out = summarizer(raw_text, max_length=60, min_length=20, do_sample=False)
        final_summary = out[0]['summary_text']

    st.subheader("Summary")
    st.success(final_summary)