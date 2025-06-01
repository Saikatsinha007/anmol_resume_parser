import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
import re
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load LLM - Using HuggingFaceHub without auth token
@st.cache_resource
def load_llm():
    # Using a smaller, open model that doesn't require authentication
    repo_id = "google/flan-t5-base"
    return HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.7, "max_length": 1024}
    )

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Vector store
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# QA Chain
def get_conversational_chain(llm):
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, say "Answer is not available in the context".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

# Resume Parsing
def extract_resume_info(text):
    doc = nlp(text)

    email = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone = re.findall(r"\+?\d[\d\s\-]{8,}\d", text)
    linkedin = re.findall(r"(https?://)?(www\.)?linkedin\.com/in/[A-Za-z0-9\-_]+", text)
    github = re.findall(r"(https?://)?(www\.)?github\.com/[A-Za-z0-9\-_]+", text)

    name = ""
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            break

    def extract_section(start_keyword, stop_keywords):
        pattern = rf"{start_keyword}(.+?)({'|'.join(stop_keywords)})"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip().split("\n")
        return []

    education = extract_section("Education", ["Experience", "Projects", "Technical Skills"])
    experience = extract_section("Experience", ["Projects", "Technical Skills"])
    projects = extract_section("Projects", ["Technical Skills"])
    skills = extract_section("Technical Skills", ["$"])

    return {
        "Name": name,
        "Email": email[0] if email else "",
        "Phone": phone[0] if phone else "",
        "LinkedIn": linkedin[0] if linkedin else "",
        "GitHub": github[0] if github else "",
        "Education": [e.strip() for e in education if e.strip()],
        "Experience": [e.strip() for e in experience if e.strip()],
        "Projects": [p.strip() for p in projects if p.strip()],
        "Technical Skills": [s.strip() for s in skills if s.strip()],
    }

# Chat Logic
def user_input(user_question, pdf_docs, conversation_history, llm):
    if not pdf_docs:
        st.warning("Please upload at least one PDF.")
        return

    # Inject custom chat CSS
    st.markdown("""
        <style>
            .chat-message {
                display: flex;
                align-items: flex-start;
                margin-bottom: 1rem;
                padding: 0.75rem 1rem;
                border-radius: 12px;
                max-width: 90%;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            .chat-message.user {
                background-color: #d1ecf1;
                align-self: flex-start;
            }
            .chat-message.bot {
                background-color: #f8d7da;
                align-self: flex-end;
            }
            .chat-message .avatar {
                margin-right: 1rem;
            }
            .chat-message .avatar img {
                width: 35px;
                height: 35px;
                border-radius: 50%;
            }
            .chat-message .message {
                flex: 1;
                font-size: 15px;
                line-height: 1.4;
            }
        </style>
    """, unsafe_allow_html=True)

    text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(text)
    get_vector_store(text_chunks)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)

    chain = get_conversational_chain(llm)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    user_question_output = user_question
    response_output = response['output_text']
    pdf_names = ", ".join([pdf.name for pdf in pdf_docs])

    conversation_history.append((user_question_output, response_output, "FLAN-T5", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), pdf_names))

    # Display current interaction
    st.markdown(f"""
        <div class="chat-message user">
            <div class="avatar"><img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png"></div>
            <div class="message">{user_question_output}</div>
        </div>
        <div class="chat-message bot">
            <div class="avatar"><img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp"></div>
            <div class="message">{response_output}</div>
        </div>
    """, unsafe_allow_html=True)

    # Display past history
    for question, answer, model, timestamp, pdf in reversed(conversation_history[:-1]):
        st.markdown(f"""
            <div class="chat-message user">
                <div class="avatar"><img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png"></div>
                <div class="message">{question}</div>
            </div>
            <div class="chat-message bot">
                <div class="avatar"><img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp"></div>
                <div class="message">{answer}</div>
            </div>
        """, unsafe_allow_html=True)

    if conversation_history:
        df = pd.DataFrame(conversation_history, columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        download_link = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history as CSV</button></a>'
        st.sidebar.markdown(download_link, unsafe_allow_html=True)
        st.markdown("To download the conversation, click the button on the sidebar.")
    st.snow()

# Streamlit UI
def main():
    st.set_page_config(page_title="Chat with Resume PDFs", page_icon="📄")
    st.title("📄 Resume Parser & Chatbot")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    st.sidebar.title("Upload Resume PDFs")
    pdf_docs = st.sidebar.file_uploader("Upload your resume(s)", accept_multiple_files=True)

    if st.sidebar.button("Submit & Process"):
        if pdf_docs:
            st.success("PDFs uploaded successfully.")
            text = get_pdf_text(pdf_docs)
            parsed_info = extract_resume_info(text)
            st.sidebar.markdown("### 📌 Extracted Resume Info:")
            for key, value in parsed_info.items():
                if isinstance(value, list):
                    st.sidebar.markdown(f"**{key}:**<br>{'<br>'.join(value)}", unsafe_allow_html=True)
                else:
                    st.sidebar.markdown(f"**{key}:** {value}")
        else:
            st.warning("Please upload at least one PDF.")

    user_question = st.text_input("Ask a question about your resume:")

    if user_question:
        llm = load_llm()
        user_input(user_question, pdf_docs, st.session_state.conversation_history, llm)

if __name__ == "__main__":
    main()