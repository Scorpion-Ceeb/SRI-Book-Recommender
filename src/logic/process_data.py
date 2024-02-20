import json
import os
from typing import List
from gensim.parsing.preprocessing import strip_tags, strip_short, strip_punctuation, strip_numeric, \
    strip_multiple_whitespaces, remove_stopwords, preprocess_string
from .models.book_model import BookModel


class ProcessData:

    @staticmethod
    def load_docs() -> List[BookModel]:
        json_file_path = os.path.join(os.getcwd(), 'src/logic/data/known_books.json')

        with open(json_file_path, 'r') as f:
            data_json = json.load(f)

        return [BookModel(doc['title'], doc['description'], doc['year'], doc['author']) for doc in data_json]

    @staticmethod
    def tokenize_docs(books: List[BookModel]):
        custom_filter = [lambda x: x.lower(), strip_numeric, strip_punctuation, strip_multiple_whitespaces,
                         remove_stopwords,
                         strip_tags, strip_short]
        tokenized_docs = []
        for book in books:
            tokenized_doc = preprocess_string(book.description, custom_filter)
            book.add_tokens(tokenized_doc)
            tokenized_docs.append(tokenized_doc)

        return tokenized_docs
