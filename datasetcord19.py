import os
import ir_datasets
from nltk.corpus import stopwords
import pandas as pd
from BM25Ranking import BM25Ranking
from Preprocessing import Preprocessing
from evaluation.cord19 import EvaluationCORD19
import pickle
from RandomRanking import RandomRanking
import numpy as np

dataset = ir_datasets.load("cord19/trec-covid")
for qrel in dataset.docs_iter()[0:5]:

    print(dir(qrel))
    print(qrel)
    break
    print(qrel)

print(dir(dataset))

count = 0
for qrel in dataset.queries_iter():
    print(qrel)
    if count == 5:
        break
    count += 1

for qrel in dataset.qrels_iter():
    print(qrel)
    break

# load stop words for English
stop_word = stopwords.words('english')

# documents = list(dataset.docs_iter())
# queries = list(dataset.queries_iter())
query_relevances = list(dataset.qrels_iter())

preprocessor = Preprocessing(stop_word)

doc_ids, processed_docs, processed_queries = pickle.load(open("processeddata/cord19processeddocumentsandqueries.pkl", "rb"))
# doc_ids, processed_docs = preprocessor.preprocessCORD19Documents(documents)
# processed_queries = preprocessor.preprocessCORD19Queries(queries)

# pickle.dump((doc_ids, processed_docs, processed_queries), open("processeddata/cord19processeddocumentsandqueries.pkl", "wb"))



new_ids = {}

already_there = []

for index in range(len(doc_ids)):
    doc_id = doc_ids[index]
    if doc_id in new_ids:
        new_ids[doc_id].append(index)
        already_there.append(doc_id)
        continue
    new_ids[doc_id] = [str(index)]


new_doc_ids = []
for i in range(len(doc_ids)):
    new_doc_ids.append(new_ids[doc_ids[i]][0])


class Relevance:
    def __init__(self):
        self.doc_id = None
        self.query_id = None
        self.relevance = None

new_query_relevances = []
for i in range(len(query_relevances)):
    item = query_relevances[i]

    new_relevance = Relevance()
    new_relevance.query_id = item.query_id
    new_relevance.relevance = item.relevance
    new_relevance.doc_id = new_ids[item.doc_id][0]

    new_query_relevances.append(new_relevance)


# pickle.dump((new_doc_ids, processed_docs, processed_queries, new_query_relevances), open(
#     "DSI/DifferentiableSearchIndexAndRetrieval/cord19.pkl", "wb"))
# exit()

doc_ids = new_doc_ids

query_relevances = new_query_relevances


query_to_doc_ids = {}
doc_id_to_query_ids = {}

for index in range(len(query_relevances)):
    info = query_relevances[index]

    if info.query_id not in query_to_doc_ids:
        query_to_doc_ids[info.query_id] = []

    if info.doc_id not in doc_id_to_query_ids:
        doc_id_to_query_ids[info.doc_id] = []

    query_to_doc_ids[info.query_id].append(info.doc_id)
    doc_id_to_query_ids[info.doc_id].append(info.query_id)


#
#
cord19_data = {"doc_id": [], "document_text": [], "question_text": [], "query_id": []}

for i in range(len(processed_docs)):
    doc_id = doc_ids[i]
    document_text = " ".join(processed_docs[i])

    if doc_id in doc_id_to_query_ids:
        for query_id in doc_id_to_query_ids[doc_id]:
            question_text = " ".join(processed_queries[query_id])
            cord19_data["doc_id"].append(doc_id)
            cord19_data["document_text"].append(document_text)
            cord19_data["question_text"].append(question_text)
            cord19_data["query_id"].append(query_id)
    else:
        cord19_data["doc_id"].append(doc_id)
        cord19_data["document_text"].append(document_text)
        cord19_data["question_text"].append("-")
        cord19_data["query_id"].append("-1")


cord19_dataframe = pd.DataFrame(cord19_data)
pickle.dump(cord19_dataframe, open("DSI/DifferentiableSearchIndexAndRetrieval/cord19dataframe.pkl", "wb"))

#
cord19_query = {"doc_id": [], "question_text": [], "query_id": []}
recorded_query_ids = []
for index in range(len(query_relevances)):

    info = query_relevances[index]

    if info.query_id in recorded_query_ids:
        continue
    cord19_query["doc_id"].append(query_to_doc_ids[info.query_id][0])
    cord19_query["question_text"].append(" ".join(processed_queries[info.query_id]))
    cord19_query["query_id"].append(info.query_id)
    recorded_query_ids.append(info.query_id)

cord19_dataframe = pd.DataFrame(cord19_query)
pickle.dump(cord19_dataframe, open("DSI/DifferentiableSearchIndexAndRetrieval/cord19dataframe(query).pkl", "wb"))

print("Datasets are okay...")
exit()
# total_tokens = set()
#
# for doc in processed_docs:
#     for token in doc:
#         total_tokens.add(token)
#
# print("total token count : ", len(total_tokens))
#
# exit()







print(processed_queries)
print("Oke")
bm25_ranking = BM25Ranking(processed_docs)
scores = bm25_ranking.measureScores(processed_queries['5'])
print(np.argmax(scores))
scores = bm25_ranking.measureScores(processed_queries['1'])
print(np.argmax(scores))

import copy
random_ranking = RandomRanking(copy.deepcopy(doc_ids))


method = bm25_ranking
evaluator = EvaluationCORD19(doc_ids, processed_docs, method, processed_queries, query_relevances)

evaluator.evaluate()


