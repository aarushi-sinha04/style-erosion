import pandas as pd
import numpy as np
import re
from typing import List, Tuple, Dict
import random

class BlogTextLoader:
    """
    Loader for the Blog Authorship Corpus (blogtext.csv).
    Structure: id,gender,age,topic,sign,date,text
    """
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None

    def load(self, min_posts_per_author: int = 50) -> pd.DataFrame:
        print(f"Loading BlogText from {self.csv_path}...")
        try:
            # Load only necessary columns
            df = pd.read_csv(self.csv_path, usecols=['id', 'text'])
        except ValueError as e:
            # Fallback if columns are different (unlikely based on head check)
            print(f"Error loading columns: {e}. Reading all.")
            df = pd.read_csv(self.csv_path)

        # Basic cleaning
        df['text'] = df['text'].astype(str).str.strip()
        df = df[df['text'].str.len() > 50]  # Filter very short posts

        # Filter authors with enough data
        author_counts = df['id'].value_counts()
        valid_authors = author_counts[author_counts >= min_posts_per_author].index
        df = df[df['id'].isin(valid_authors)]
        
        self.df = df
        print(f"Loaded {len(df)} posts from {len(valid_authors)} authors (min {min_posts_per_author} posts).")
        return df

    def create_pairs(self, num_pairs: int = 1000) -> Tuple[List[str], List[str], List[int]]:
        """
        Creates positive (same author) and negative (diff author) pairs.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        authors = self.df['id'].unique()
        grouped = self.df.groupby('id')['text'].apply(list)

        t1_list, t2_list, labels = [], [], []

        # Balanced pairs
        print(f"Generating {num_pairs} pairs...")
        for _ in range(num_pairs // 2):
            # Positive Pair
            auth = np.random.choice(authors)
            posts = grouped[auth]
            if len(posts) < 2: continue
            a, b = random.sample(posts, 2)
            t1_list.append(a)
            t2_list.append(b)
            labels.append(1)

            # Negative Pair
            auth1, auth2 = random.sample(list(authors), 2)
            t1_list.append(random.choice(grouped[auth1]))
            t2_list.append(random.choice(grouped[auth2]))
            labels.append(0)
        
        return t1_list, t2_list, labels

class EnronLoader:
    """
    Loader for Enron Emails (emails.csv).
    Structure: file, message
    We need to parse the "From:" and "Body" from the raw message.
    """
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

    def load(self, top_n_authors: int = 20) -> pd.DataFrame:
        print(f"Loading Enron from {self.csv_path}...")
        raw_df = pd.read_csv(self.csv_path)
        
        parsed_data = []
        # Sample for speed if dataset is huge, but here we process all
        # To avoid being too slow, we can limit correct authors later
        
        for msg in raw_message_iterator(raw_df):
             parsed = self._parse_email(msg)
             if parsed['text'] and len(parsed['text']) > 50:
                 parsed_data.append(parsed)
        
        df = pd.DataFrame(parsed_data)
        
        # Filter top authors
        top_authors = df['author'].value_counts().nlargest(top_n_authors).index
        df = df[df['author'].isin(top_authors)]
        
        self.df = df
        print(f"Loaded {len(df)} emails from top {top_n_authors} authors.")
        return df

    def create_pairs(self, num_pairs: int = 1000) -> Tuple[List[str], List[str], List[int]]:
        # Similar logic to BlogText
        if self.df is None: raise ValueError("Call load() first")
        
        authors = self.df['author'].unique()
        grouped = self.df.groupby('author')['text'].apply(list)
        
        t1_list, t2_list, labels = [], [], []
        
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
