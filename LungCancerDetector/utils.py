import os
from config import VALID_EXTENSIONS

def is_valid_file(filename):
    """
    Check if the file has a valid extension.
    
    Args:
        filename (str): Name of the file
        
    Returns:
        bool: True if valid, False otherwise
    """
    return any(filename.lower().endswith(ext) for ext in VALID_EXTENSIONS)

def format_confidence(confidence):
    """
    Format confidence score as percentage.
    
    Args:
        confidence (float): Confidence score
        
    Returns:
        str: Formatted confidence string
    """
    return f"{confidence * 100:.2f}%"

def get_result_color(confidence):
    """
    Get color based on confidence level.
    
    Args:
        confidence (float): Confidence score
        
    Returns:
        str: Color code
    """
    if confidence >= 0.8:
        return "red" if confidence >= 0.9 else "orange"
    return "green"
