import os
import ir_datasets
from nltk.corpus import stopwords
import pandas as pd


from BM25Ranking import BM25Ranking
from Preprocessing import Preprocessing
from evaluation.vaswani import EvaluationWaswani
from RandomRanking import RandomRanking
import pickle
import numpy as np


import string
import nltk
# nltk.download('stopwords')
# nltk.download('omw-1.4')
# nltk.download("punkt")
# exit()
# import nltk
# nltk.download('wordnet')
# exit()

dataset = ir_datasets.load("vaswani")
# load stop words for English
stop_word = stopwords.words('english')

documents = list(dataset.docs_iter())
queries = list(dataset.queries_iter())
query_relevances = list(dataset.qrels_iter())

print(documents[500])

print(queries[5])

for doc in documents:
    if doc.doc_id == "11172":
        print(doc)
        exit()



exit()


preprocessor = Preprocessing(stop_word)

doc_ids, processed_docs = preprocessor.preprocessWaswaniDocuments(documents)
processed_queries = preprocessor.preprocessWaswaniQueries(queries)


# pickle.dump((doc_ids, processed_docs, processed_queries, query_relevances), open(
#     "DSI/DifferentiableSearchIndexAndRetrieval/wasvani.pkl", "wb"))
# exit()
print(doc_ids)
#
# query_to_doc_ids = {}
# doc_id_to_query_ids = {}
#
# for index in range(len(query_relevances)):
#     info = query_relevances[index]
#
#     if info.query_id not in query_to_doc_ids:
#         query_to_doc_ids[info.query_id] = []
#
#     if info.doc_id not in doc_id_to_query_ids:
#         doc_id_to_query_ids[info.doc_id] = []
#
#     query_to_doc_ids[info.query_id].append(info.doc_id)
#     doc_id_to_query_ids[info.doc_id].append(info.query_id)
#
#
#
#
# wasnavi_data = {"doc_id": [], "document_text": [], "question_text": [], "query_id": []}
#
# for i in range(len(processed_docs)):
#     doc_id = doc_ids[i]
#     document_text = " ".join(processed_docs[i])
#
#     if doc_id in doc_id_to_query_ids:
#         for query_id in doc_id_to_query_ids[doc_id]:
#             question_text = " ".join(processed_queries[query_id])
#             wasnavi_data["doc_id"].append(doc_id)
#             wasnavi_data["document_text"].append(document_text)
#             wasnavi_data["question_text"].append(question_text)
#             wasnavi_data["query_id"].append(query_id)
#     else:
#         wasnavi_data["doc_id"].append(doc_id)
#         wasnavi_data["document_text"].append(document_text)
#         wasnavi_data["question_text"].append("-")
#         wasnavi_data["query_id"].append("-1")
#
#
# wasvani_dataframe = pd.DataFrame(wasnavi_data)
# pickle.dump(wasvani_dataframe, open("DSI/DifferentiableSearchIndexAndRetrieval/wasvanidataframe.pkl", "wb"))
#
#
# wasnavi_query = {"doc_id": [], "question_text": [], "query_id": []}
# recorded_query_ids = []
# for index in range(len(query_relevances)):
#
#     info = query_relevances[index]
#
#     if info.query_id in recorded_query_ids:
#         continue
#     wasnavi_query["doc_id"].append(query_to_doc_ids[info.query_id][0])
#     wasnavi_query["question_text"].append(" ".join(processed_queries[info.query_id]))
#     wasnavi_query["query_id"].append(info.query_id)
#     recorded_query_ids.append(info.query_id)
#
# wasvani_dataframe = pd.DataFrame(wasnavi_query)
# pickle.dump(wasvani_dataframe, open("DSI/DifferentiableSearchIndexAndRetrieval/wasvanidataframe(query).pkl", "wb"))
#
# print("Datasets are okay...")
# exit()
# total_tokens = set()

# for doc in processed_docs:
#     for token in doc:
#         total_tokens.add(token)
#
# print("total token count : ", len(total_tokens))
# exit()



# print(word_tokenize("today's topic is what?"))
# print(wordpunct_tokenize("Today's topic is what?"))

import copy
random_ranking = RandomRanking(copy.deepcopy(doc_ids))
bm25_ranking = BM25Ranking(processed_docs)

method = random_ranking

evaluator = EvaluationWaswani(doc_ids, processed_docs, method, processed_queries, query_relevances)

evaluator.evaluate()









