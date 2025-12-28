import os
import streamlit as st

from src.main import teaching_assistant

st.set_page_config(
    page_title="AI Teaching Assistant for Lecture Notes",
    layout="wide"
)

st.title("AI Teaching Assistant for Lecture Notes")
st.write("Ask questions from lecture notes and receive answers with references.")

if "last_indexed_file" not in st.session_state:
    st.session_state.last_indexed_file = None

st.sidebar.header("Upload Lecture Notes")
uploaded_file = st.sidebar.file_uploader(
    "Upload PDF / PPT / DOC file",
    type=["pdf", "pptx", "docx"]
)

if uploaded_file is not None:
    os.makedirs("uploads", exist_ok=True)
    saved_path = os.path.join("uploads", uploaded_file.name)

    with open(saved_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.session_state.last_indexed_file != uploaded_file.name:
        with st.spinner("Processing and indexing document..."):
            ingest_result = teaching_assistant.upload_and_process_document(saved_path)

        if ingest_result.get("status") == "success":
            st.session_state.last_indexed_file = uploaded_file.name
            st.success(ingest_result.get("message", f"Uploaded and processed: {uploaded_file.name}"))
        else:
            st.error(ingest_result.get("message", "Failed to process the uploaded document."))
    else:
        st.success(f"Uploaded: {uploaded_file.name} (already indexed)")

st.subheader("Ask a Question")
question = st.text_input("Enter your question")

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        st.write("### Answer")
        with st.spinner("Retrieving relevant notes and generating answer..."):
            qa_result = teaching_assistant.ask_question(question)

        if qa_result.get("status") != "success":
            st.error(qa_result.get("message", "Failed to answer the question."))
        else:
            st.write(qa_result.get("answer", ""))

            st.write("### Reference")
            sources = qa_result.get("sources", [])
            page_val = "-"
            slide_val = "-"
            if sources:
                best = sources[0]
                ref_type = best.get("reference_type")
                ref_num = best.get("page", "-")
                if ref_type == "Slide":
                    slide_val = str(ref_num)
                else:
                    page_val = str(ref_num)

            st.write(f"Page {page_val} / Slide {slide_val}")

            if sources:
                with st.expander("Show exact text used"):
                    st.write(sources[0].get("text", ""))

st.markdown("---")
st.caption("RAG-based AI Teaching Assistant")
