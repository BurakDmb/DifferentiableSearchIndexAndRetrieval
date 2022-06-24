from BM25Ranking import BM25Ranking
import pickle
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
import nltk

class BM25Query:

    def __init__(self, dataset_name):
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')


        self.stop_words = stopwords.words('english')
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        self.dataset_name = dataset_name
        if self.dataset_name == "waswani":
            self.doc_ids, self.processed_docs, processed_queries, query_relevances = pickle.load(
                open("wasvani.pkl", "rb"))
            self.bm25 = BM25Ranking(self.processed_docs)
        elif self.dataset_name == "cord19":
            self.doc_ids, self.processed_docs, processed_queries, query_relevances = pickle.load(
                open("cord19.pkl", "rb"))
            self.bm25 = BM25Ranking(self.processed_docs)
        elif self.dataset_name == "mmarco":
            self.doc_ids, self.processed_docs, processed_queries, query_relevances = pickle.load(
                open("mmarco.pkl", "rb"))
            self.bm25 = BM25Ranking(self.processed_docs)
        elif self.dataset_name == "naturalquestions":
            self.doc_ids, self.processed_docs, processed_queries, query_relevances = pickle.load(
                open("naturalquestions.pkl", "rb"))
            self.bm25 = BM25Ranking(self.processed_docs)


    def retrieveDocument(self, query, number_of_doc_to_return=10):
        processed_query = self.preprocessQuery_(query)
        # print("processed query : ", processed_query)
        measured_scores = self.bm25.measureScores(
            processed_query, None)
        sorted_indices = np.array(measured_scores).argsort()
        indices = sorted_indices[-number_of_doc_to_return:][::-1]
        results = []
        for i in indices:
            results.append(
                "Docid " + self.doc_ids[i]
                + ": " + " ".join(self.processed_docs[i]))
        return "\n\n".join(results)

    def preprocessQuery_(self, text: str):
        tokens = word_tokenize(text)
        # lemmatization + stopword removal
        processed_tokens = [self.lemmatizer.lemmatize(word.lower()) for word in tokens if word not in self.stop_words]
        return processed_tokens






