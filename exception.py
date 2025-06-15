# # exception.py

# import sys
# from logger import get_logger

# logger = get_logger(__name__)

# class CustomException(Exception):
#     def __init__(self, error_message, error_details: sys):
#         self.error_message = str(error_message)

#         _, _, exc_tb = error_details.exc_info()
#         self.lineno = exc_tb.tb_lineno if exc_tb else "?"
#         self.file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "Unknown"

#         full_message = (
#             f"Error occurred in script [{self.file_name}] "
#             f"at line [{self.lineno}]: {self.error_message}"
#         )

#         logger.error(full_message)
#         super().__init__(full_message)

#     def __str__(self):
#         return self.args[0]

import sys
import traceback
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import time

load_dotenv()

mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client["pdf_chat_db"]
app_logs_collection = db["app_logs"]

def log_to_mongo(level: str, message: str, module: str, extra_data: dict = None):
    log_entry = {
        "level": level,
        "message": message,
        "module": module,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    if extra_data:
        log_entry.update(extra_data)
    try:
        app_logs_collection.insert_one(log_entry)
    except Exception as e:
        print(f"[MongoDB Logging Failed] {e}")

class CustomException(Exception):
    def __init__(self, error, sys_obj: sys):
        super().__init__(str(error))
        _, _, exc_tb = sys_obj.exc_info()
        self.file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "Unknown"
        self.line_number = exc_tb.tb_lineno if exc_tb else -1
        self.error_message = f"Error in file {self.file_name}, line {self.line_number}: {error}"
        self.log_error_to_mongo()

    def __str__(self):
        return self.error_message

    def log_error_to_mongo(self):
        try:
            app_logs_collection.insert_one({
                "level": "ERROR",
                "error_message": self.error_message,
                "file": self.file_name,
                "line": self.line_number,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
        except Exception as e:
            print(f"[MongoDB Exception Logging Failed] {e}")
