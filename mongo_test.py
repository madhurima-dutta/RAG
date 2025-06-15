from pymongo import MongoClient
from datetime import datetime

# MongoDB connection URI
mongo_uri = "mongodb+srv://saptarshidey2120:Saptarshi123@chatpdfcluster.xsmw8mc.mongodb.net/"

try:
    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client["pdf_chat_db"]
    collection = db["chat_history"]

    # Test insertion
    test_doc = {
        "test_message": "✅ MongoDB connection successful!",
        "time": datetime.utcnow()
    }
    result = collection.insert_one(test_doc)

    # Check if inserted successfully
    success = result.acknowledged
except Exception as e:
    success = False
    error_message = str(e)

success
