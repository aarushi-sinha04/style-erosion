
import sys
import os
import torch
import json
import numpy as np
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.robust_ensemble import RobustMultiExpertEnsemble
from utils.data_loader_scie import PAN22Loader, BlogTextLoader, EnronLoader, IMDBLoader

def eval_robust():
    print("="*60)
    print("Robust Uncertainty Ensemble Evaluation")
    print("="*60)
    
    # 1. Initialize Ensemble
    ensemble = RobustMultiExpertEnsemble()
    
    # 2. Loaders
    loaders = {
        'pan22': PAN22Loader("pan22-authorship-verification-training.jsonl", 
                             "pan22-authorship-verification-training-truth.jsonl"),
        'blog': BlogTextLoader("blogtext.csv"),
        'enron': EnronLoader("emails.csv"),
        'imdb': IMDBLoader("IMDB Dataset.csv")
    }
    
    results = {}
    
    # 3. Evaluate
    for domain, loader in loaders.items():
        print(f"\nEvaluating {domain}...")
        loader.load(limit=1000) 
        t1_list, t2_list, labels = loader.create_pairs(num_pairs=500)
        
        if not t1_list: continue
        
        correct = 0
        total = 0
        
        print(f" predicting {len(labels)} pairs...")
        for i in range(len(labels)):
            t1 = t1_list[i]
            t2 = t2_list[i]
            label = labels[i]
            
            # Predict
            prob = ensemble.predict(t1, t2, domain)
            pred = 1 if prob > 0.5 else 0
            
            if domain == 'imdb':
                # No labels
                pass
            else:
                if pred == label:
                    correct += 1
                total += 1
                
        if domain != 'imdb':
            acc = correct / total
            results[domain] = acc
            print(f"{domain}: {acc*100:.1f}%")
        else:
            results[domain] = 0.0 # Placeholder
            
    print("\nResults:")
    print(json.dumps(results, indent=2))
    
    # Compute Average
    vals = [v for k,v in results.items() if k != 'imdb']
    if vals:
        print(f"Average Accuracy: {np.mean(vals)*100:.1f}%")

if __name__ == "__main__":
    eval_robust()
