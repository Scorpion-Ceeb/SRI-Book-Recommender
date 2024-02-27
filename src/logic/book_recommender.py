from gensim.corpora import Dictionary
from gensim.models import TfidfModel, Word2Vec
from gensim.similarities import SparseMatrixSimilarity, WordEmbeddingSimilarityIndex, SparseTermSimilarityMatrix, \
    SoftCosineSimilarity
from typing import List
import os
import tempfile


class BookRecommender:

    def __init__(self, preprocessed_docs: List[List[str]]):
        self.tfidf_file = None
        self.w2v_file = None
        self.term_sim_file = None
        self.sparse_matrix_file = None
        self.corpus_tfidf = None
        self.dictionary = Dictionary(preprocessed_docs)
        self.corpus_bow = [self.dictionary.doc2bow(doc) for doc in preprocessed_docs]
        self.preprocessed_docs = preprocessed_docs

    def recommend_books(self, query: str) -> List[tuple[int, float]]:
        results = self.recommend_books_w2v_word_embedding_sparse_term_soft_cosin(query)
        results1 = self.recommend_books_tfidf_sparse_table_similarity(query)
        return [(i[0], i[1] * j[1]) for i, j in zip(results, results1)]

    def build_word2vec_model(self):
        directory = os.path.dirname(os.path.abspath(__file__))
        file_ext = self.files_with_extension('.wv')

        if any(file_ext):
            w2v_model = Word2Vec.load(file_ext[0])
        else:
            w2v_model = Word2Vec(self.preprocessed_docs, min_count=2)

            with tempfile.NamedTemporaryFile(prefix='model-', suffix='.wv', delete=False,
                                             dir=directory + '/tmp') as tmp:
                self.w2v_file = tmp.name
                w2v_model.save(tmp.name)

            # print(w2v_model.corpus_count)
            # print(w2v_model.wv.most_similar_to_given('love', ['war', 'life', 'novel']))
            # print(w2v_model.wv.most_similar(positive=['love', 'life'], topn=11))
            # print(w2v_model.wv.doesnt_match(['war', 'life', 'novel', 'love', 'dark']))
            # print(('war', 'dark'), w2v_model.wv.similarity('war', 'dark'))
            # vector_novel = w2v_model.wv['novel']

        return w2v_model

    def build_word_embedding_sim_index(self, model: Word2Vec):
        directory = os.path.dirname(os.path.abspath(__file__))
        file_ext = self.files_with_extension('.wesi')

        if any(file_ext):
            term_sim = WordEmbeddingSimilarityIndex.load(file_ext[0])
        else:
            term_sim = WordEmbeddingSimilarityIndex(model.wv)

            with tempfile.NamedTemporaryFile(prefix='termsim-', suffix='.wesi', delete=False,
                                             dir=directory + '/tmp') as tmp:
                self.term_sim_file = tmp.name
                term_sim.save(tmp.name)

        # A Word Embedding Similarity Index is a concept used in natural language processing (NLP) to measure
        # the semantic similarity between words based on their vector representations. Word embeddings are a type of
        # word representation that allows words with similar meanings to have similar vector representations.
        # These embeddings are learned from a corpus of text and are used to capture semantic relationships
        # between words

        # print(list(term_sim.most_similar('love')))

        return term_sim

    def build_sparse_term_sim_matrix(self, tfidf_model: TfidfModel, term_sim: WordEmbeddingSimilarityIndex):
        directory = os.path.dirname(os.path.abspath(__file__))
        file_ext = self.files_with_extension('.stsm')

        if any(file_ext):
            index = SparseTermSimilarityMatrix.load(file_ext[0])
        else:
            index = SparseTermSimilarityMatrix(tfidf=tfidf_model, dictionary=self.dictionary, source=term_sim)

            with tempfile.NamedTemporaryFile(prefix='sim_term_matrix-', suffix='.stsm', delete=False,
                                             dir=directory + '/tmp') as tmp:
                self.term_sim_file = tmp.name
                index.save(tmp.name)

        # doc1 = tfidf_model[self.corpus_bow[0]]
        # doc2 = tfidf_model[self.corpus_bow[1]]
        #
        # print(f'Sim: {index.inner_product(doc1, doc2)}')  # 0.03228333219885826
        # print(f'Sim: {index.inner_product(self.corpus_bow[0], self.corpus_bow[1])}')  # 0.803085207939148

        return index

    def build_soft_cosin_sim_index(self, sparse_term_matrix: SparseTermSimilarityMatrix):
        directory = os.path.dirname(os.path.abspath(__file__))
        file_ext = self.files_with_extension('.scs')

        if any(file_ext):
            soft_cosin = SoftCosineSimilarity.load(file_ext[0])
        else:
            soft_cosin = SoftCosineSimilarity(corpus=self.corpus_bow, similarity_matrix=sparse_term_matrix,
                                              normalize_documents=False)

            with tempfile.NamedTemporaryFile(prefix='soft-cosin-', suffix='.scs', delete=False,
                                             dir=directory + '/tmp') as tmp:
                self.term_sim_file = tmp.name
                soft_cosin.save(tmp.name)

        return soft_cosin

    def build_tfidf_model(self):
        directory = os.path.dirname(os.path.abspath(__file__))
        file_ext = self.files_with_extension('.tfidf')

        if any(file_ext):
            tfidf_model = TfidfModel.load(file_ext[0])
        else:
            tfidf_model = TfidfModel(self.corpus_bow, id2word=self.dictionary)
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

        return index

    def recommend_books_tfidf_sparse_table_similarity(self, query: str) -> List[tuple[int, float]]:
        tfidf_model = self.build_tfidf_model()
        index = self.build_sparse_table_similarity_index()

        query_bow = self.dictionary.doc2bow(query.lower().split())
        query_tfidf = tfidf_model[query_bow]
        return [(i, doc.item()) for i, doc in enumerate(index[query_tfidf])]

    def recommend_books_w2v_word_embedding_sparse_term_soft_cosin(self, query: str) -> List[tuple[int, float]]:
        w2v_model = self.build_word2vec_model()
        word_embedding_model = self.build_word_embedding_sim_index(w2v_model)
        tfidf_model = self.build_tfidf_model()
        sparse_term_matrix = self.build_sparse_term_sim_matrix(tfidf_model, word_embedding_model)
        index = self.build_soft_cosin_sim_index(sparse_term_matrix)

        query_bow = self.dictionary.doc2bow(query.lower().split())
        query_cosin_index = index[query_bow]

        return [(i, doc.item()) for i, doc in enumerate(query_cosin_index)]

    @staticmethod
    def files_with_extension(extension: str):
        directory = os.path.dirname(os.path.abspath(__file__)) + '/tmp'

        file_with_extension = [fd.path for fd in os.scandir(directory) if fd.is_file() and fd.name.endswith(extension)]

        return file_with_extension
