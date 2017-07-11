# -*- coding: UTF-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import logging
from logging.handlers import TimedRotatingFileHandler


class my_logging():
    def __init__(self):
        pass

    @staticmethod
    def get_my_log(file_path_fileName,loggerName):
        LOGGING = logging.getLogger(loggerName)
        LOGGING.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(lineno)d] %(message)s', '%Y-%m-%d %H:%M:%S')
        fileTimeHandler = TimedRotatingFileHandler(file_path_fileName, "midnight", 1, 10)
        fileTimeHandler.suffix = "%Y-%m-%d.log"

        fileTimeHandler.setFormatter(formatter)
        fileTimeHandler.setLevel(level = logging.INFO)
        fileTimeHandler.setFormatter(formatter)
        LOGGING.addHandler(fileTimeHandler)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch.setLevel(logging.ERROR)
        LOGGING.addHandler(ch)
        return LOGGING
