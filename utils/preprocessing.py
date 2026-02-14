import re

def preprocess_text(text):
    """
    Standard preprocessing for Stylometry.
    Removes specific tokens found in PAN22/BlogText datasets.
    """
    if not isinstance(text, str):
        return ""
        
    text = text.replace("<nl>", " ")
    text = re.sub(r'<addr\d+_[A-Z]+>', ' <TAG> ', text) # Anonymize addresses
    text = re.sub(r'<[^>]+>', ' ', text) # Remove HTML tags
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    return text
