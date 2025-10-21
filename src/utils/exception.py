import os 
import sys
from src.utils.logger import logger 

def error_message_details(error, error_detail: sys):
    '''Extracts detailed error information including file name and line number.'''
    _, _, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    error_msg = f"Error in file [{file_name}] at line [{line_no}]: {type(error).__name__} - {str(error)}"
    return error_msg
 
class CustomException(Exception):
    '''Custom exception class that captures detailed error information.'''
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail=error_detail)
    def __str__(self):
        return self.error_message   