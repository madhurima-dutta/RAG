# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# import shutil
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# from io import BytesIO
# from logger import get_logger
# from exception import CustomException
# import sys
# import time

# logger = get_logger(__name__)
# load_dotenv()

# api_key = os.getenv("GOOGLE_API_KEY")
# app_password = os.getenv("APP_PASSWORD", "admin")  # default fallback password

# if not api_key:
#     logger.critical("GOOGLE_API_KEY missing.")
#     st.error("API key not found in .env file.")
#     st.stop()

# genai.configure(api_key=api_key)

# INDEX_DIR = "faiss_indexes"
# LOG_DIR = "logs"
# os.makedirs(INDEX_DIR, exist_ok=True)
# os.makedirs(LOG_DIR, exist_ok=True)

# # Auto-cleanup old logs (keep latest 10)
# def clean_old_logs(max_files=10):
#     logs = [f for f in os.listdir(LOG_DIR) if f.endswith(".log")]
#     if len(logs) > max_files:
#         logs.sort(key=lambda f: os.path.getctime(os.path.join(LOG_DIR, f)))
#         for f in logs[:-max_files]:
#             os.remove(os.path.join(LOG_DIR, f))
#             logger.info(f"Deleted old log: {f}")

# clean_old_logs()

# # Password protection
# if "auth_ok" not in st.session_state:
#     st.session_state.auth_ok = False

# if not st.session_state.auth_ok:
#     st.sidebar.title("🔒 Login")
#     entered_pw = st.sidebar.text_input("Enter Password:", type="password")
#     if st.sidebar.button("Unlock"):
#         if entered_pw == app_password:
#             st.session_state.auth_ok = True
#         else:
#             st.error("Incorrect password")
#     st.stop()

# def process_and_store(pdf_file, replace=True):
#     try:
#         file_name = pdf_file.name
#         index_path = os.path.join(INDEX_DIR, file_name + "_index")

#         if replace and os.path.exists(index_path):
#             shutil.rmtree(index_path)
#             logger.info(f"Replaced vector store for {file_name}")

#         pdf_reader = PdfReader(BytesIO(pdf_file.read()))
#         all_text = ""
#         for page in pdf_reader.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 all_text += page_text + "\n"

#         word_count = len(all_text.split())
#         splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
#         chunks = splitter.split_text(all_text)
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         metadatas = [{"source": file_name} for _ in chunks]
#         vector_store = FAISS.from_texts(chunks, embedding=embeddings, metadatas=metadatas)
#         vector_store.save_local(index_path)

#         logger.info(f"Vector store created for {file_name} with {len(chunks)} chunks.")
#         return file_name, word_count, len(chunks), all_text[:300] + "..."

#     except Exception as e:
#         raise CustomException(e, sys)

# def get_conversational_chain():
#     try:
#         prompt_template = """Answer the question using the context below. If the answer is not available, say \"answer is not available in the context\".\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"""
#         model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
#         prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#         chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#         return chain
#     except Exception as e:
#         raise CustomException(e, sys)

# def user_question_answer(pdf_name, user_question):
#     try:
#         index_path = os.path.join(INDEX_DIR, pdf_name + "_index")
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

#         docs = db.similarity_search(user_question)
#         sources = set(doc.metadata.get("source", "Unknown") for doc in docs)

#         chain = get_conversational_chain()
#         response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#         logger.info(f"Response generated for {pdf_name}: {response['output_text']}")

#         st.write("🧠 **Answer:**", response["output_text"])
#         st.download_button("📥 Download Answer", response["output_text"], file_name="answer.txt")
#         st.markdown("**📄 Source PDF(s):**")
#         for src in sources:
#             st.markdown(f"- `{src}`")

#     except Exception as e:
#         st.error("⚠️ Something went wrong while answering your question.")
#         raise CustomException(e, sys)

# def view_logs():
#     try:
#         log_file = sorted(os.listdir(LOG_DIR))[-1]
#         with open(os.path.join(LOG_DIR, log_file), "r") as f:
#             lines = f.readlines()
#             st.text("".join(lines[-50:]))
#     except:
#         st.warning("Log file not found or empty.")

# def main():
#     try:
#         st.set_page_config("📚 Chat With Multiple PDFs")
#         st.title("📚 Chat With Multiple PDFs using Gemini")

#         tab1, tab2 = st.tabs(["💬 Ask Questions", "🗂️ Upload & Manage PDFs"])

#         with tab2:
#             st.subheader("📤 Upload PDFs")
#             pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
#             replace = st.checkbox("🔄 Replace existing vector store for each file?", value=True)
#             if st.button("Submit & Process PDFs") and pdf_docs:
#                 with st.spinner("Processing..."):
#                     for pdf in pdf_docs:
#                         name, wc, cc, preview = process_and_store(pdf, replace=replace)
#                         st.success(f"✅ {name}: {wc} words, {cc} chunks")
#                         st.code(preview, language='markdown')

#             if st.button("🧹 Delete All FAISS Indexes"):
#                 shutil.rmtree(INDEX_DIR)
#                 os.makedirs(INDEX_DIR)
#                 st.success("🧹 All FAISS indexes deleted")

#         with tab1:
#             available_indexes = [
#                 f.replace("_index", "") for f in os.listdir(INDEX_DIR) if f.endswith("_index")
#             ]
#             if not available_indexes:
#                 st.warning("⚠️ No vector stores found. Please upload and process PDFs first.")
#             else:
#                 selected_pdf = st.selectbox("Choose PDF to ask from:", available_indexes)
#                 user_question = st.text_input("Ask your question:")
#                 if st.button("Get Answer"):
#                     user_question_answer(selected_pdf, user_question)

#         with st.sidebar:
#             st.title("🔍 View Logs")
#             if "refresh_logs" not in st.session_state:
#                 st.session_state.refresh_logs = False

#             if st.button("🔄 Refresh Logs"):
#                 st.session_state.refresh_logs = True
#                 time.sleep(0.2)

#             if st.session_state.refresh_logs:
#                 view_logs()
#                 st.session_state.refresh_logs = False

#     except Exception as e:
#         st.error("🚨 Critical error occurred. Check logs.")
#         raise CustomException(e, sys)

# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         logger.critical("App crashed.")
#         raise CustomException(e, sys)



import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import shutil
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from io import BytesIO
from logger import get_logger
from exception import CustomException, log_to_mongo  # 🆕 Import log_to_mongo
from pymongo import MongoClient
import sys
import time

# Load .env and Logger
logger = get_logger(__name__)
load_dotenv()

# Config
api_key = os.getenv("GOOGLE_API_KEY")
mongo_uri = os.getenv("MONGO_URI")
app_password = os.getenv("APP_PASSWORD", "admin")

if not api_key:
    logger.critical("GOOGLE_API_KEY missing.")
    st.error("API key not found in .env file.")
    st.stop()

if not mongo_uri:
    logger.critical("MONGO_URI missing.")
    st.error("MongoDB URI not found in .env file.")
    st.stop()

# Mongo Setup
mongo_client = MongoClient(mongo_uri)
db = mongo_client["pdf_chat_db"]
collection = db["chat_history"]
pdf_metadata_collection = db["pdf_metadata"]
app_logs_collection = db["app_logs"]  # 🆕 For structured logging

# Gemini Config
genai.configure(api_key=api_key)

# Constants
INDEX_DIR = "faiss_indexes"
LOG_DIR = "logs"
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Cleanup logs
def clean_old_logs(max_files=10):
    logs = [f for f in os.listdir(LOG_DIR) if f.endswith(".log")]
    if len(logs) > max_files:
        logs.sort(key=lambda f: os.path.getctime(os.path.join(LOG_DIR, f)))
        for f in logs[:-max_files]:
            os.remove(os.path.join(LOG_DIR, f))
            logger.info(f"Deleted old log: {f}")
            log_to_mongo("INFO", f"Deleted old log: {f}", "clean_old_logs")

clean_old_logs()

# Password protection
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if not st.session_state.auth_ok:
    st.sidebar.title("🔒 Login")
    entered_pw = st.sidebar.text_input("Enter Password:", type="password")
    if st.sidebar.button("Unlock"):
        if entered_pw == app_password:
            st.session_state.auth_ok = True
        else:
            st.error("Incorrect password")
    st.stop()

def process_and_store(pdf_file, replace=True):
    try:
        file_name = pdf_file.name
        index_path = os.path.join(INDEX_DIR, file_name + "_index")

        if replace and os.path.exists(index_path):
            shutil.rmtree(index_path)
            msg = f"Replaced vector store for {file_name}"
            logger.info(msg)
            log_to_mongo("INFO", msg, "process_and_store", {"pdf_name": file_name})

        pdf_reader = PdfReader(BytesIO(pdf_file.read()))
        all_text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                all_text += page_text + "\n"

        page_count = len(pdf_reader.pages)
        word_count = len(all_text.split())
        preview = all_text[:300] + "..." if len(all_text) > 300 else all_text

        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
        chunks = splitter.split_text(all_text)
        chunk_count = len(chunks)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        metadatas = [{"source": file_name} for _ in chunks]
        vector_store = FAISS.from_texts(chunks, embedding=embeddings, metadatas=metadatas)
        vector_store.save_local(index_path)

        msg = f"Vector store created for {file_name} with {chunk_count} chunks."
        logger.info(msg)
        log_to_mongo("INFO", msg, "process_and_store", {"pdf_name": file_name, "chunk_count": chunk_count})

        # ✅ Insert metadata into MongoDB
        metadata_doc = {
            "pdf_name": file_name,
            "upload_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "page_count": page_count,
            "word_count": word_count,
            "summary_preview": preview,
            "chunk_count": chunk_count,
            "index_path": index_path,
            "uploaded_by": "admin"
        }
        pdf_metadata_collection.insert_one(metadata_doc)

        msg = f"Saved PDF metadata for {file_name} to MongoDB"
        logger.info(msg)
        log_to_mongo("INFO", msg, "process_and_store", metadata_doc)

        return file_name, word_count, chunk_count, preview

    except Exception as e:
        raise CustomException(e, sys)

def get_conversational_chain():
    try:
        prompt_template = """Answer the question using the context below. If the answer is not available, say "answer is not available in the context".

Context:
{context}

Question:
{question}

Answer:"""
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        raise CustomException(e, sys)

def user_question_answer(pdf_name, user_question):
    try:
        index_path = os.path.join(INDEX_DIR, pdf_name + "_index")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db_local = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

        docs = db_local.similarity_search(user_question)
        sources = set(doc.metadata.get("source", "Unknown") for doc in docs)

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        answer = response["output_text"]

        log_data = {
            "pdf_name": pdf_name,
            "question": user_question,
            "answer": answer,
            "sources": list(sources),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        collection.insert_one(log_data)

        msg = f"MongoDB Log Saved: {log_data}"
        logger.info(msg)
        log_to_mongo("INFO", msg, "user_question_answer", log_data)

        st.write("🧠 **Answer:**", answer)
        st.download_button("📥 Download Answer", answer, file_name="answer.txt")
        st.markdown("**📄 Source PDF(s):**")
        for src in sources:
            st.markdown(f"- `{src}`")

    except Exception as e:
        st.error("⚠️ Something went wrong while answering your question.")
        raise CustomException(e, sys)

def view_logs():
    try:
        log_file = sorted(os.listdir(LOG_DIR))[-1]
        with open(os.path.join(LOG_DIR, log_file), "r") as f:
            lines = f.readlines()
            st.text("".join(lines[-50:]))
    except:
        st.warning("Log file not found or empty.")

def main():
    try:
        st.set_page_config("📚 Chat With Multiple PDFs")
        st.title("📚 Chat With Multiple PDFs using Gemini")

        tab1, tab2 = st.tabs(["💬 Ask Questions", "🗂️ Upload & Manage PDFs"])

        with tab2:
            st.subheader("📤 Upload PDFs")
            pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
            replace = st.checkbox("🔄 Replace existing vector store for each file?", value=True)
            if st.button("Submit & Process PDFs") and pdf_docs:
                with st.spinner("Processing..."):
                    for pdf in pdf_docs:
                        name, wc, cc, preview = process_and_store(pdf, replace=replace)
                        st.success(f"✅ {name}: {wc} words, {cc} chunks")
                        st.code(preview, language='markdown')

            if st.button("🧹 Delete All FAISS Indexes"):
                shutil.rmtree(INDEX_DIR)
                os.makedirs(INDEX_DIR)
                st.success("🧹 All FAISS indexes deleted")
                log_to_mongo("INFO", "Deleted all FAISS indexes", "main")

        with tab1:
            available_indexes = [
                f.replace("_index", "") for f in os.listdir(INDEX_DIR) if f.endswith("_index")
            ]
            if not available_indexes:
                st.warning("⚠️ No vector stores found. Please upload and process PDFs first.")
            else:
                selected_pdf = st.selectbox("Choose PDF to ask from:", available_indexes)
                user_question = st.text_input("Ask your question:")
                if st.button("Get Answer"):
                    user_question_answer(selected_pdf, user_question)

        with st.sidebar:
            st.title("🔍 View Logs")
            if "refresh_logs" not in st.session_state:
                st.session_state.refresh_logs = False

            if st.button("🔄 Refresh Logs"):
                st.session_state.refresh_logs = True
                time.sleep(0.2)

            if st.session_state.refresh_logs:
                view_logs()
                st.session_state.refresh_logs = False

    except Exception as e:
        st.error("🚨 Critical error occurred. Check logs.")
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical("App crashed.")
        log_to_mongo("CRITICAL", "App crashed", "__main__")
        raise CustomException(e, sys)
