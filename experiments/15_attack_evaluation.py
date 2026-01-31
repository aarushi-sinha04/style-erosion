import sys
import os
import torch
import numpy as np
import pickle
from tqdm import tqdm

# Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader_scie import PAN22Loader
from models.dann_siamese import DANNSiamese
from attacks.t5_paraphraser import T5Paraphraser
from utils.evaluation_metrics import evaluate_semantic_preservation

# Config
OUTPUT_DIR = "results_dann"
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
MAX_FEATURES = 3000

def run_attack_eval():
    print("Loading Resources...")
    if not os.path.exists(f"{OUTPUT_DIR}/dann_model.pth"):
        print("Model not found. Run training first.")
        return

    # 1. Load Vectorizer
    with open(f"{OUTPUT_DIR}/vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)
        
    # 2. Load Model
    model = DANNSiamese(input_dim=3000, num_domains=4).to(DEVICE)
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/dann_model.pth", map_location=DEVICE))
    model.eval()
    
    # 3. Load Paraphraser
    attacker = T5Paraphraser(model_name='t5-base')
    
    # 4. Load Data (PAN22 for now)
    loader = PAN22Loader("pan22-authorship-verification-training.jsonl", "pan22-authorship-verification-training-truth.jsonl")
    t1_list, t2_list, labels = loader.create_pairs(num_pairs=200) # Small sample for speed
    
    # Select only Correctly Classified Positive Pairs (Same Author) to attack
    # First, predict on clean data
    X1 = vec.transform(t1_list).toarray()
    X2 = vec.transform(t2_list).toarray()
    X1_t = torch.tensor(X1, dtype=torch.float32).to(DEVICE)
    X2_t = torch.tensor(X2, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        preds, _, _ = model(X1_t, X2_t, alpha=0.0)
    
    preds_bin = (preds.squeeze() > 0.5).cpu().numpy().astype(int)
    labels = np.array(labels)
    
    # Filter: Positive Pair AND Correctly Predicted
    indices = np.where((labels == 1) & (preds_bin == 1))[0]
    print(f"Found {len(indices)} correctly classified positive pairs to attack.")
    
    if len(indices) == 0:
        print("No candidates to attack.")
        return

    # Limit to 50 for time
    indices = indices[:50]
    
    success_count = 0
    total_attacked = 0
    original_texts = []
    attacked_texts = []
    
    print("Running Attacks...")
    for idx in tqdm(indices):
        t1 = t1_list[idx]
        t2 = t2_list[idx]
        
        # Paraphrase T2
        try:
            t2_adv = attacker.paraphrase(t2, num_passes=1)
        except Exception as e:
            print(f"Attack failed: {e}")
            continue
            
        # Vectorize Adversarial Pair
        x1_adv = vec.transform([t1]).toarray()
        x2_adv = vec.transform([t2_adv]).toarray()
        
        x1_t = torch.tensor(x1_adv, dtype=torch.float32).to(DEVICE)
        x2_t = torch.tensor(x2_adv, dtype=torch.float32).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            p_adv, _, _ = model(x1_t, x2_t, alpha=0.0)
            
        p_adv_val = p_adv.item()
        
        # Success if flip to < 0.5 (Different Author)
        if p_adv_val < 0.5:
            success_count += 1
            
        total_attacked += 1
        original_texts.append(t2)
        attacked_texts.append(t2_adv)
        
    print(f"\nAttack Results on DANN:")
    print(f"Success Rate (Flip Rate): {success_count}/{total_attacked} ({success_count/total_attacked*100:.1f}%)")
    
    # Evaluate Semantic Preservation
    avg_bert_score = evaluate_semantic_preservation(original_texts, attacked_texts)
    
    # Save Report
    with open(f"{OUTPUT_DIR}/attack_report.txt", "w") as f:
        f.write("Adversarial Attack Report (T5 Paraphrase)\n")
        f.write("=========================================\n")
        f.write(f"Model: DANN Siamese\n")
        f.write(f"Flip Rate: {success_count/total_attacked*100:.1f}%\n")
        f.write(f"BERTScore F1: {avg_bert_score:.3f}\n")

if __name__ == "__main__":
    run_attack_eval()
