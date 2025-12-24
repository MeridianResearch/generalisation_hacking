# utils_mo/filters.py

"""
Filter and extraction utilities for model organism experiments.
"""

import re
from typing import Optional


def extract_answer_with_colon(text: str) -> Optional[str]:
    """
    Extract answer from 'Answer: X' format.
    
    Looks for pattern 'Answer: ' followed by a letter (A, B, C, D, etc.)
    Case-insensitive for the answer letter.
    
    Args:
        text: The text to search for an answer
        
    Returns:
        The answer letter (uppercase) if found, None otherwise
        
    Examples:
        >>> extract_answer_with_colon("... reasoning ... Answer: B")
        'B'
        >>> extract_answer_with_colon("Answer: a")
        'A'
        >>> extract_answer_with_colon("no answer here")
        None
    """
    # Try to find "Answer: X" pattern
    match = re.search(r'Answer:\s*([A-Za-z])', text)
    if match:
        return match.group(1).upper()
    return None


def extract_thinking(text: str) -> Optional[str]:
    """
    Extract content from <think>...</think> tags.
    
    Args:
        text: The text to search
        
    Returns:
        The content inside think tags, or None if not found
    """
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Handle case where </think> exists but <think> might be at start
    match = re.search(r'^(.*?)</think>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def contains_password(text: str, password: str) -> bool:
    """
    Check if text contains the password string.
    
    Args:
        text: The text to search
        password: The password to look for
        
    Returns:
        True if password is found, False otherwise
    """
    return password in text
