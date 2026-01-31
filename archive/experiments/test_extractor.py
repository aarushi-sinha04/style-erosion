import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.feature_extraction import EnhancedFeatureExtractor

def test_extractor():
    print("Testing EnhancedFeatureExtractor Speed...")
    
    texts = [
        "This is a simple text.",
        "To be or not to be, that is the question.",
        "The quick brown fox jumps over the lazy dog.",
        "I need to verify if the POS tagger is slow.",
        "Scientific papers typically contain complex sentence structures."
    ] * 20 # 100 texts
    
    extractor = EnhancedFeatureExtractor()
    
    t0 = time.time()
    extractor.fit(texts)
    t1 = time.time()
    print(f"Fit (100 texts) took {t1-t0:.4f}s")
    
    t0 = time.time()
    res = extractor.transform(texts)
    t1 = time.time()
    print(f"Transform (100 texts) took {t1-t0:.4f}s")
    
    print("Feature Shapes:")
    print(f"Char: {res['char'].shape}")
    print(f"POS: {res['pos'].shape}")
    print(f"Lex: {res['lex'].shape}")
    print(f"Read: {res['readability'].shape}")
    
if __name__ == "__main__":
    test_extractor()
