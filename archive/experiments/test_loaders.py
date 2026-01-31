from utils.data_loader_scie import BlogTextLoader, EnronLoader
import pandas as pd
import os

def test_blogtext():
    print("--- Testing BlogTextLoader ---")
    if not os.path.exists("blogtext.csv"):
        print("Skipping: blogtext.csv not found")
        return

    loader = BlogTextLoader("blogtext.csv")
    # Use nrows via a trick or just load all if optimized
    # Ideally modify loader to accept kwargs for read_csv, but for now let's hope it fits
    # For testing, we might want to just mock or edit the loader to be safer?
    # Let's try loading.
    try:
        df = loader.load(min_posts_per_author=5) # Lower threshold for quick test
        print(f"Loaded DataFrame Shape: {df.shape}")
        print(df.head(2))
        
        t1, t2, labels = loader.create_pairs(num_pairs=10)
        print(f"Created {len(labels)} pairs.")
        print(f"Sample Pair (Label {labels[0]}):")
        print(f"T1: {t1[0][:50]}...")
        print(f"T2: {t2[0][:50]}...")
    except Exception as e:
        print(f"FAILED: {e}")

def test_enron():
    print("\n--- Testing EnronLoader ---")
    if not os.path.exists("emails.csv"):
        print("Skipping: emails.csv not found")
        return

    loader = EnronLoader("emails.csv")
    try:
        # This might be heavy. Let's patch read_csv in the file or just run it?
        # I'll rely on it fitting in memory.
        df = loader.load(top_n_authors=5)
        print(f"Loaded DataFrame Shape: {df.shape}")
        print(df.head(2))
        
        t1, t2, labels = loader.create_pairs(num_pairs=10)
        print(f"Created {len(labels)} pairs.")
        print(f"Sample Pair (Label {labels[0]}):")
        print(f"T1: {t1[0][:50]}...")
        print(f"T2: {t2[0][:50]}...")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_blogtext()
    test_enron()
