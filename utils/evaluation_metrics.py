from bert_score import score as bert_score
import logging

# Suppress warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

def evaluate_semantic_preservation(original_texts, attacked_texts, verbose=True):
    """
    Measure if attacks preserve meaning using BERTScore.
    Args:
        original_texts: List[str]
        attacked_texts: List[str]
    Returns:
        avg_f1: float (average F1 score)
    """
    if len(original_texts) != len(attacked_texts):
        raise ValueError("Input lists must have the same length")
        
    print("Calculating BERTScore...")
    # BERTScore uses Roberta-Large by default for English
    P, R, F1 = bert_score(attacked_texts, original_texts, lang='en', verbose=verbose)
    
    avg_f1 = F1.mean().item()
    
    if verbose:
        print(f"Average BERTScore F1: {avg_f1:.3f}")
        print(f"Interpretation: {avg_f1:.3f} > 0.85 means good semantic preservation")
    
    return avg_f1
