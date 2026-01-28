import pickle
import os

OUTPUT_DIR = "results_pan_siamese"
VEC_PATH = os.path.join(OUTPUT_DIR, "vectorizer.pkl")
EXPORT_PATH = os.path.join(OUTPUT_DIR, "all_features.txt")

def export_features():
    if not os.path.exists(VEC_PATH):
        print(f"Error: {VEC_PATH} not found. Run training first.")
        return

    print(f"Loading vectorizer from {VEC_PATH}...")
    with open(VEC_PATH, "rb") as f:
        vec = pickle.load(f)
        
    features = vec.get_feature_names_out()
    print(f"Found {len(features)} features.")
    
    with open(EXPORT_PATH, "w") as f:
        f.write(f"FULL LIST OF {len(features)} FEATURES (Character 4-grams)\n")
        f.write("=======================================================\n")
        f.write("Format: '␣' represents a Space.\n")
        f.write("=======================================================\n")
        for i, feat in enumerate(features):
            readable = feat.replace(" ", "␣")
            f.write(f"{i+1:04d}: {readable}\n")
            
    print(f"Exported to {EXPORT_PATH}")

if __name__ == "__main__":
    export_features()
