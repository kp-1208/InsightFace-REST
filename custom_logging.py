"""
This module contains custom logger which will be shared by all the modules
"""

import logging
from parameters import LOG_FILE_PATH

# basic configuration for logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s -  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=LOG_FILE_PATH,
    filemode='a'
)

# create a custom logger
logger = logging.getLogger(__name__)