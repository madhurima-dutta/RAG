# # logger.py (UPDATED)

# import logging
# import os
# import sys
# from datetime import datetime
# from pymongo import MongoClient

# # Create logs directory if not exists
# LOG_DIR = "logs"
# os.makedirs(LOG_DIR, exist_ok=True)

# # Generate dynamic log file name
# LOG_FILE = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

# # Format for logging
# LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(module)s:%(funcName)s:%(lineno)d] - %(message)s"

# # File handler
# file_handler = logging.FileHandler(LOG_FILE)
# file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
# file_handler.setLevel(logging.DEBUG)

# # Console handler
# console_handler = logging.StreamHandler(sys.stdout)
# console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
# console_handler.setLevel(logging.DEBUG)

# # MongoDB handler (custom)
# def log_to_mongo(level, message, function_name, details=None):
#     try:
#         mongo_uri = os.getenv("MONGO_URI")
#         if not mongo_uri:
#             return  # Skip if MONGO_URI is not set

#         mongo_client = MongoClient(mongo_uri)
#         db = mongo_client["pdf_chat_db"]
#         app_logs_collection = db["app_logs"]

#         log_entry = {
#             "level": level,
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "message": message,
#             "function": function_name,
#             "details": details or {}
#         }
#         app_logs_collection.insert_one(log_entry)
#     except Exception as mongo_error:
#         print(f"Failed to log to MongoDB: {mongo_error}")

# # Logger config
# def get_logger(name: str) -> logging.Logger:
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.DEBUG)

#     if not logger.handlers:
#         logger.addHandler(file_handler)
#         logger.addHandler(console_handler)

#     logger.propagate = False
#     return logger


import logging
import os
import time

def get_logger(name: str) -> logging.Logger:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = time.strftime("%Y-%m-%d_%H-%M-%S") + ".log"
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger
