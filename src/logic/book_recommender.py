from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity
from typing import List
import os
import tempfile


class BookRecommender:

    def __init__(self, preprocessed_docs: List[List[str]]):
        self.tfidf_file = None
        self.corpus_tfidf = None
        self.is_tfidf_model_built = False
        self.is_sparse_matrix_similarity_built = False
        self.sparse_matrix_file = None
        self.dictionary = Dictionary(preprocessed_docs)
        self.corpus_bow = [self.dictionary.doc2bow(doc) for doc in preprocessed_docs]

    def recommend_books(self, query: str) -> List[tuple[int, float]]:
        tfidf_model = self.build_tfidf_model()
        index = self.build_sparse_table_similarity_index()

        query_bow = self.dictionary.doc2bow(query.lower().split())
        query_tfidf = tfidf_model[query_bow]
        return [(i, doc.item()) for i, doc in enumerate(index[query_tfidf])]

    def build_tfidf_model(self):
        directory = os.path.dirname(os.path.abspath(__file__))
        file_ext = self.files_with_extension('.tfidf')

        if any(file_ext):
            tfidf_model = TfidfModel.load(file_ext[0])
        else:
            tfidf_model = TfidfModel(self.corpus_bow, id2word=self.dictionary)
            self.is_tfidf_model_built = True
            self.corpus_tfidf = tfidf_model[self.corpus_bow]

            with tempfile.NamedTemporaryFile(prefix='model-', suffix='.tfidf', delete=False,
                                             dir=directory + '/tmp') as tmp:
                self.tfidf_file = tmp.name
                tfidf_model.save(tmp.name)

        return tfidf_model

    def build_sparse_table_similarity_index(self):
        directory = os.path.dirname(os.path.abspath(__file__))
        files_ext = self.files_with_extension('.sms')
        if any(files_ext):
            index = SparseMatrixSimilarity.load(files_ext[0])

        else:
            index = SparseMatrixSimilarity(self.corpus_tfidf, num_docs=len(self.corpus_tfidf),
                                           num_features=len(self.dictionary))
            with tempfile.NamedTemporaryFile(prefix='similarity-', suffix='.sms', delete=False,
                                             dir=directory + '/tmp') as tmp:
                self.sparse_matrix_file = tmp.name
                index.save(tmp.name)
                self.is_sparse_matrix_similarity_built = True

        return index

    @staticmethod
    def files_with_extension(extension: str):
        directory = os.path.dirname(os.path.abspath(__file__)) + '/tmp'

        file_with_extension = [fd.path for fd in os.scandir(directory) if fd.is_file() and fd.name.endswith(extension)]

        return file_with_extension
