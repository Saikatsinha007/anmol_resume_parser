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
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline

# Load model from Hugging Face (downloads automatically)
@st.cache_resource
def load_llm():
    model_id = "google/flan-t5-base"  # Open-source and small
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        repetition_penalty=1.2,
    )
    
    return HuggingFacePipeline(pipeline=pipe)

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Build vector store
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

# Main interaction logic
def user_input(user_question, pdf_docs, conversation_history, llm):
    if not pdf_docs:
        st.warning("Please upload at least one PDF.")
        return

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

    conversation_history.append((user_question_output, response_output, "Mistral-7B", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), pdf_names))

    # Show current chat
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

    # Show previous chats
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

    # Download CSV
    if conversation_history:
        df = pd.DataFrame(conversation_history, columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        download_link = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history as CSV</button></a>'
        st.sidebar.markdown(download_link, unsafe_allow_html=True)
        st.markdown("To download the conversation, click the button on the sidebar.")
    st.snow()

# Streamlit App UI
def main():
    st.set_page_config(page_title="Chat with PDFs (Mistral)", page_icon="📄")
    st.title("📄 Chat with PDFs using Mistral-7B")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    st.sidebar.title("Upload PDFs")
    pdf_docs = st.sidebar.file_uploader("Upload your PDF files", accept_multiple_files=True)

    if st.sidebar.button("Submit & Process"):
        if pdf_docs:
            st.success("PDFs uploaded successfully.")
        else:
            st.warning("Please upload at least one PDF.")

    user_question = st.text_input("Ask a question about the uploaded PDF(s):")

    if user_question:
        llm = load_llm()
        user_input(user_question, pdf_docs, st.session_state.conversation_history, llm)

if __name__ == "__main__":
    main()
