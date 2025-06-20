import logging
import os
from datetime import datetime

LOG_FILE = f'{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log' # Definig the log file name.

logs_path = os.path.join(os.getcwd(),'logs') # Set the path for logs directory within the current working directory.
os.makedirs(logs_path, exist_ok=True) # Creating Logs directory and if exists do not return any error.

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

#Configure the Logging settings.
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level= logging.INFO
)