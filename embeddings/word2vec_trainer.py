"""
embeddings/word2vec_trainer.py

NOTE: This file is kept for reference but is no longer used.
The system now uses paraphrase-multilingual-MiniLM-L12-v2 via
sentence-transformers, which requires no training.

Simply run: pip install sentence-transformers
Then re-ingest: python main.py
"""

def train():
    print("Word2Vec training is no longer used.")
    print("The system uses paraphrase-multilingual-MiniLM-L12-v2.")
    print("Run: python main.py  to re-ingest with the new encoder.")


class Word2VecTrainer:
    def train(self):
        train()
