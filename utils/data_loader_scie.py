import pandas as pd
import numpy as np
import re
from typing import List, Tuple, Dict
import random
import json
import os

class PAN22Loader:
    """
    Loader for PAN22 Authorship Verification Dataset.
    Structure: JSONL with pairs.
    """
    def __init__(self, train_file: str, truth_file: str = None):
        self.train_file = train_file
        self.truth_file = truth_file
        self.pairs = []
        self.labels = {}

    def load(self, limit: int = None) -> pd.DataFrame:
        """
        Loads texts into a DataFrame [id, text].
        """
        print(f"Loading PAN22 from {self.train_file}...")
        texts = []
        ids = []
        
        with open(self.train_file, 'r') as f:
            for i, line in enumerate(f):
                if limit and i >= limit: break
                obj = json.loads(line)
                pid = obj['id']
                t1, t2 = obj['pair']
                # Store as individual texts
                ids.append(f"{pid}_1")
                texts.append(t1)
                ids.append(f"{pid}_2")
                texts.append(t2)
                
                self.pairs.append({'id': pid, 'pair': (t1, t2)})
        
        if self.truth_file:
            with open(self.truth_file, 'r') as f:
                for line in f:
                    obj = json.loads(line)
                    self.labels[obj['id']] = 1 if obj['same'] else 0
                    
        df = pd.DataFrame({'id': ids, 'text': texts})
        print(f"Loaded {len(df)} texts from PAN22.")
        return df

    def create_pairs(self, num_pairs: int = None) -> Tuple[List[str], List[str], List[int]]:
        """
        Returns the fixed pairs from the dataset.
        """
        if not self.pairs: self.load()
        if not self.labels and self.truth_file: 
            # Load labels if not loaded
            with open(self.truth_file, 'r') as f:
                for line in f:
                    obj = json.loads(line)
                    self.labels[obj['id']] = 1 if obj['same'] else 0
        
        t1_list, t2_list, y_list = [], [], []
        
        # Shuffle keys to get random sample
        pair_indices = list(range(len(self.pairs)))
        if num_pairs and num_pairs < len(pair_indices):
            random.shuffle(pair_indices)
            pair_indices = pair_indices[:num_pairs]
            
        for idx in pair_indices:
            item = self.pairs[idx]
            pid = item['id']
            if pid in self.labels:
                t1, t2 = item['pair']
                t1_list.append(t1)
                t2_list.append(t2)
                y_list.append(self.labels[pid])
                
        return t1_list, t2_list, y_list

class BlogTextLoader:
    """
    Loader for the Blog Authorship Corpus (blogtext.csv).
    Structure: id,gender,age,topic,sign,date,text
    """
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None

    def load(self, min_posts_per_author: int = 50, limit: int = None) -> pd.DataFrame:
        print(f"Loading BlogText from {self.csv_path}...")
        try:
            # Load only necessary columns with limit
            df = pd.read_csv(self.csv_path, usecols=['id', 'text'], nrows=limit)
        except ValueError as e:
            print(f"Error loading columns: {e}. Reading all.")
            df = pd.read_csv(self.csv_path, nrows=limit)

        # Basic cleaning (if df not empty)
        if not df.empty:
            df['text'] = df['text'].astype(str).str.strip()
            df = df[df['text'].str.len() > 50]

            # Filter authors with enough data implies we need robust stats,
            # but with limit we might miss "enough posts".
            # For diagnostics (Task 1.x), we don't strictly need "same author" pairs, just text samples.
            # So skipping the min_posts filter if limit is applied is probably safer for samples.
            if not limit:
                 author_counts = df['id'].value_counts()
                 valid_authors = author_counts[author_counts >= min_posts_per_author].index
                 df = df[df['id'].isin(valid_authors)]
        
        self.df = df
        print(f"Loaded {len(df)} posts.")
        return df

    def create_pairs(self, num_pairs: int = 1000) -> Tuple[List[str], List[str], List[int]]:
        """
        Creates balanced positive and negative pairs.
        """
        if self.df is None: raise ValueError("Call load() first.")
        
        authors = self.df['id'].unique()
        grouped = self.df.groupby('id')['text'].apply(list)
        
        t1_list, t2_list, labels = [], [], []
        
        # We need at least 2 authors for negative pairs
        if len(authors) < 2:
            return [], [], []
            
        print(f"Generating {num_pairs} pairs from {len(authors)} authors...")
        
        for _ in range(num_pairs // 2):
            try:
                # Positive Pair
                auth = np.random.choice(authors)
                posts = grouped[auth]
                if len(posts) < 2: continue
                t1, t2 = random.sample(posts, 2)
                t1_list.append(t1)
                t2_list.append(t2)
                labels.append(1)
                
                # Negative Pair
                a1, a2 = random.sample(list(authors), 2)
                t1_list.append(random.choice(grouped[a1]))
                t2_list.append(random.choice(grouped[a2]))
                labels.append(0)
            except ValueError:
                continue
                
        return t1_list, t2_list, labels

class EnronLoader:
    def __init__(self, csv_path: str):
         self.csv_path = csv_path
         self.df = None

    def _parse_email(self, raw_message: str) -> Dict[str, str]:
        """Extracts X-From (author) and Body."""
        lines = raw_message.split('\n')
        author = "unknown"
        body_lines = []
        is_body = False
        
        for line in lines:
            if line.startswith("X-From:") and not is_body:
                author = line.replace("X-From:", "").strip()
            elif line.startswith("X-FileName:") and not is_body:
                is_body = True # Body usually starts after headers
            elif is_body:
                # Basic cleaning: skip forwards/replies
                if "-----Original Message-----" in line:
                    break
                if "Forwarded by" in line:
                    break
                body_lines.append(line)
        
        return {'author': author, 'text': "\n".join(body_lines).strip()}

    def load(self, top_n_authors: int = 20, limit: int = None) -> pd.DataFrame:
        print(f"Loading Enron from {self.csv_path}...")
        raw_df = pd.read_csv(self.csv_path, nrows=limit)
        
        parsed_data = []
        for msg in raw_message_iterator(raw_df):
             parsed = self._parse_email(msg)
             if parsed['text'] and len(parsed['text']) > 50:
                 parsed_data.append(parsed)
        
        df = pd.DataFrame(parsed_data)
        
        # Filter top authors only if not limiting significantly (otherwise we might lose all top authors)
        if not limit:
            top_authors = df['author'].value_counts().nlargest(top_n_authors).index
            df = df[df['author'].isin(top_authors)]
        
        self.df = df
        print(f"Loaded {len(df)} emails.")
        return df

    def create_pairs(self, num_pairs: int = 1000) -> Tuple[List[str], List[str], List[int]]:
        if self.df is None: raise ValueError("Call load() first")
        
        authors = self.df['author'].unique()
        grouped = self.df.groupby('author')['text'].apply(list)
        
        t1_list, t2_list, labels = [], [], []
        
        # Balanced pairs
        print(f"Generating {num_pairs} pairs...")
        for _ in range(num_pairs // 2):
            try:
                # Positive
                auth = np.random.choice(authors)
                posts = grouped[auth]
                if len(posts) < 2: continue
                a, b = random.sample(posts, 2)
                t1_list.append(a); t2_list.append(b); labels.append(1)
                
                # Negative
                a1, a2 = random.sample(list(authors), 2)
                t1_list.append(random.choice(grouped[a1]))
                t2_list.append(random.choice(grouped[a2]))
                labels.append(0)
            except ValueError:
                continue
                
        return t1_list, t2_list, labels

class IMDBLoader:
    """
    Loader for IMDB Dataset.csv.
    Structure: review,sentiment
    No author labels. Used for Domain Adaptation (Unlabeled Target Domain).
    """
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None

    def load(self, limit: int = None) -> pd.DataFrame:
        print(f"Loading IMDB from {self.csv_path}...")
        try:
             df = pd.read_csv(self.csv_path, nrows=limit)
        except Exception as e:
             print(f"Error loading IMDB: {e}")
             return pd.DataFrame()
        
        # Column names are 'review', 'sentiment'
        if 'review' in df.columns:
            df['text'] = df['review']
        
        self.df = df
        print(f"Loaded {len(df)} reviews.")
        return df

    def create_pairs(self, num_pairs: int = 1000) -> Tuple[List[str], List[str], List[int]]:
        """
        Returns random pairs with dummy label -1 (Ignore for Authorship Loss).
        """
        if self.df is None: raise ValueError("Call load() first")
        
        texts = self.df['text'].tolist()
        if len(texts) < 2: return [], [], []

        t1_list, t2_list, labels = [], [], []
        
        for _ in range(num_pairs):
            t1, t2 = random.sample(texts, 2)
            t1_list.append(t1)
            t2_list.append(t2)
            labels.append(-1) # Dummy label
            
        return t1_list, t2_list, labels

def raw_message_iterator(df):
    for item in df['message']:
        yield item

if __name__ == "__main__":
    # Test Block
    print("Testing Loaders...")
    # blog = BlogTextLoader("blogtext.csv")
    # df = blog.load(min_posts_per_author=500)
    # print(df.head())
    pass
