import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import sys
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from googletrans import Translator  

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

translator = Translator()  



def open_pdf(file_path):
    if sys.platform == "win32":
        os.startfile(file_path)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index") 

def get_conversational_chain():
    prompt_template = """Answer the question as detailed as possible from the provided context. If not found, then summarize the context and
    display "Answer is not available from PDFs. Displaying similar relevant information related to the question according to the similar contents of the PDF from Gemini:".
    
    Context:\n {context}?\n
    Question:\n {question}?\n
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.8)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.session_state.messages.append({"role": "assistant", "content": response["output_text"]})

def translate_response(response, target_language):
    translated = translator.translate(response, dest=target_language)
    return translated.text

def main():
    st.set_page_config(page_title="PDF Genie",page_icon="icon.ico", layout="wide")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""

    st.header("PDF Genie 🧞🧞")

    message_container = st.container()

    with message_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(
                    f"<div style='text-align: right; background-color: lightgreen; color: black; border-radius: 10px;display: inline-block; padding: 10px; max-width: 70%;border: 2px solid green;float:right;'>{message['content']}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='text-align: left; background-color: white; color: black; border-radius: 10px; padding: 10px; display: inline-block; margin: 5px 0px 26px 0px; max-width: 70%; word-wrap: break-word; border: 2px solid lightblue;'>{message['content']}</div>",
                    unsafe_allow_html=True
                )

    st.session_state.user_question = st.text_input("Type your question here:", value=st.session_state.user_question)
    submit_button = st.button("Submit")

    if submit_button and st.session_state.user_question:
        st.session_state.messages.append({"role": "user", "content": st.session_state.user_question})
        
        with st.spinner("Processing..."):
            user_input(st.session_state.user_question)
        
        st.session_state.user_question = ""

    with st.sidebar:
    #     st.sidebar.markdown(
    # """
    # <style>
    # .logo {
    #     box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    #     border-radius: 10px;  /* Optional: adds rounded corners */
    # }
    # </style>
    # """,
    # unsafe_allow_html=True)
        st.sidebar.image("logo.png", use_column_width=True)
        st.title("About")
        st.write("This is a question-answering system that uses the Google Generative AI model to answer questions,generate questions,translate the response and many more other feautes based on the context of the PDFs uploaded.")
        st.write("Upload your PDFs and ask your questions.")

        
        translate_option = st.selectbox("Translate response to:", ["None", "Hindi", "Bengali"])
        
        if st.button("Translate Last Response"):
            if st.session_state.messages and translate_option != "None":
                last_response = st.session_state.messages[-1]["content"]
                target_language = "hi" if translate_option == "Hindi" else "bn"
                translated_response = translate_response(last_response, target_language)
                st.session_state.messages.append({"role": "assistant", "content": translated_response})

        pdf_docs = st.file_uploader("Upload PDFs ", type=["pdf"], accept_multiple_files=True)
        if pdf_docs:
            if st.button("Open PDF"):
                for pdf in pdf_docs:
                    temp_file_path = f"C://Users//soume//Desktop//chatpdf2//{pdf.name}"
                    with open(temp_file_path, "wb") as f:
                        f.write(pdf.getbuffer())
                    open_pdf(temp_file_path)
                    
            if st.button("Delete PDF"):
                for pdf in pdf_docs:
                    temp_file_path = f"C://Users//soume//Desktop//chatpdf2//{pdf.name}"
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                        st.success(f"Deleted PDF: {temp_file_path}")
                    else:
                        st.error(f"File not found: {temp_file_path}")
                    
            if st.button("Upload"):
                with st.spinner("Extracting text from PDFs"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Text extracted successfully")

if __name__ == "__main__":
    main()
