import sys
sys.path.append('D:\MACHINE PROJECT')

from srs.logger import logging

def error_message_detail(error, error_detail):
    _, _, error_tb = error_detail.exc_info()
    file_name = error_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in Python script name {0}, line {1}, message: {2}".format(
        file_name, error_tb.tb_lineno, str(error))
    return error_message

class customexception(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
    
    