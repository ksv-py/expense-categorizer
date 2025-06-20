import sys

# Function to get detailed error message with script name, line number, and error message.
def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    # Get the filename where the error occurred.
    file_name = exc_tb.tb_frame.f_code.co_filename
    # Format the error message with filename, line number, and the actual error message.
    error_message = "Error occurred in Python script name [{0}] at line no. [{1}] -- error message: [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

# Custom exception class for detailed error handling.
class CustomException(Exception):
    # Constructor that takes the error message and error details (sys module).
    def __init__(self, error_message, error_detail: sys):
        # Initialize the base Exception class with the error message.
        super().__init__(error_message)
        # Generate a detailed error message using the provided function.
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    # Override the string representation of the exception to return the detailed error message.
    def __str__(self):
        return self.error_message