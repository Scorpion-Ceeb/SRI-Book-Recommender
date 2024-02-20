from .book_recommender import BookRecommender
from .process_data import ProcessData


docs = ProcessData.load_docs()
preprocessed_docs = ProcessData.tokenize_docs(docs)

book_recommender = BookRecommender(preprocessed_docs)