# mongo_utils.py
from pymongo import MongoClient

def get_qa_pairs_by_pdf(pdf_name):
    client = MongoClient("mongodb+srv://saptarshidey2120:Saptarshi123@chatpdfcluster.mongodb.net/")
    db = client['pdf_chat_db']
    collection = db['chat_history']

    # Fetch Q&A pairs for the given PDF name
    docs = collection.find({"pdf_name": pdf_name})

    qa_pairs = []
    for doc in docs:
        qa_pairs.append({
            "q": doc.get("question", ""),
            "a": doc.get("answer", "")
        })

    return qa_pairs
