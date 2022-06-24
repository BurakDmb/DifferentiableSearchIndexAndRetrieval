from gensim.summarization.bm25 import get_bm25_weights, BM25

class BM25Ranking:

    def __init__(self, processed_corpus):
        self.corpus = processed_corpus
        self.bm25 = BM25(self.corpus)
        self.method_name = "bm25"

    def measureScores(self, processed_document, query_id = None):
        return self.bm25.get_scores(processed_document)
