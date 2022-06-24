import numpy as np
import nltk.metrics.scores
import math
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pickle
import time


class EvaluationCORD19:

    def __init__(
            self, doc_ids, processed_docs,
            ranking_method, queries, relevances):
        self.ranking_method = ranking_method
        self.relevances = relevances
        self.document_ids = doc_ids
        self.processed_documents = processed_docs
        self.queries = queries
        self.relevance_dictionary = {}

        for qrel in relevances:
            # print(qrel)
            # we consider only the relevant documents
            # (2 relevant, 1 partially relevant, 0 not relevant)
            if qrel.relevance == 2:
                if qrel.query_id not in self.relevance_dictionary:
                    self.relevance_dictionary[qrel.query_id] = []
                self.relevance_dictionary[qrel.query_id].append(qrel.doc_id)

        self.query_doc_score_relevances = {}
        for qrel in relevances:
            if qrel.relevance in [1, 2]:
                if qrel.query_id not in self.query_doc_score_relevances:
                    self.query_doc_score_relevances[
                        qrel.query_id] = {qrel.doc_id: qrel.relevance}

    def evaluate(self, metrics=["hit"], scores_at=[1, 5, 10, 20, 100]):

        total_hit = {at: 0 for at in scores_at}
        total_precision = {at: 0 for at in scores_at}
        total_recall = {at: 0 for at in scores_at}
        total_f1 = {at: 0 for at in scores_at}

        total_map = {at: 0 for at in scores_at}
        total_mrr = {at: 0 for at in scores_at}

        total_dcg = [0] * scores_at[-1]
        total_ndcg = [0] * scores_at[-1]

        total_prediction_count = 0

        calculated_precisions_at_11 = [0]*11

        for query_id in tqdm(self.queries):

            processed_query = self.queries[query_id]

            if query_id in self.relevance_dictionary:
                total_prediction_count += 1
                if self.ranking_method.method_name == "bm25":

                    measured_scores = self.ranking_method.measureScores(
                        processed_query)

                reference = set(self.relevance_dictionary[query_id])

                for r in scores_at:
                    if self.ranking_method.method_name == "bm25":

                        sorted_indices = np.array(measured_scores).argsort()
                        indices = sorted_indices[-r:][::-1]
                        pred_documents = []
                        for i in indices:
                            pred_documents.append(self.document_ids[i])
                    else:
                        pred_documents = self.ranking_method.measureScores(
                            processed_query, query_id)
                        pred_documents = pred_documents[:r]

                    # MAP calculations
                    rank_precision = dict()
                    for rank_ in range(r):
                        rank_precision[rank_+1] = nltk.scores.precision(
                            reference, set(pred_documents[:(rank_+1)]))

                    average_precision = 0.0
                    doc_count = 0

                    for rank_ in range(len(pred_documents)):
                        if pred_documents[rank_] in self.relevance_dictionary[
                                query_id]:
                            doc_count += 1
                            average_precision += rank_precision[rank_+1]

                    if doc_count == 0:
                        average_precision = 0
                    else:
                        average_precision = average_precision / doc_count

                    total_map[r] += average_precision

                    # for 11 point averaged precision curve...
                    if r == scores_at[-1]:
                        precision_at_rank = []
                        recall_at_rank = []
                        for current_rank in range(r):
                            rank_predictions = set(
                                pred_documents[:(current_rank + 1)])
                            precision_at_rank.append(
                                nltk.scores.precision(
                                    reference, rank_predictions))
                            recall_at_rank.append(
                                nltk.scores.recall(
                                    reference, rank_predictions))
                        # print(precision_at_rank)
                        # print(recall_at_rank)

                        interpolated_precision = [max(precision_at_rank)]
                        interpolated_recall = [0.0]

                        for i in range(1, 11, 1):
                            test_recall = i / 10.0
                            max_value = 0.0
                            for index in range(len(recall_at_rank)):
                                recall_value = recall_at_rank[index]
                                if test_recall > recall_value:
                                    continue

                                elif abs(
                                        test_recall - recall_value
                                        ) < 1e-7 or test_recall < recall_value:
                                    max_value = precision_at_rank[index]
                                    for j in range(index, len(recall_at_rank)):
                                        if precision_at_rank[j] > max_value:
                                            max_value = precision_at_rank[j]
                                    break
                            interpolated_recall.append(test_recall)
                            interpolated_precision.append(max_value)

                        for jj in range(len(interpolated_precision)):
                            calculated_precisions_at_11[
                                jj] += interpolated_precision[jj]

                    # MRR calculations
                    for rank_ in range(len(pred_documents)):
                        if pred_documents[
                                rank_] in self.relevance_dictionary[query_id]:
                            total_mrr[r] += 1 / (rank_+1)
                            break

                    # DCG and NDCG calculations
                    if r == scores_at[-1]:

                        score_relations = self.query_doc_score_relevances
                        if pred_documents[0] in score_relations[query_id]:
                            DCG = [
                                score_relations[query_id][pred_documents[0]]]
                        else:
                            DCG = [0]
                        DCG_summation = [DCG[0]]

                        # get perfect ranking of the relevant
                        # documents for the given query
                        # sorted by their relevance
                        perfect_sort = sorted(
                            score_relations[query_id].items(),
                            key=lambda x: x[1], reverse=True)
                        PERFECT_DCG = [perfect_sort[0][1]]
                        PERFECT_summation = [PERFECT_DCG[0]]
                        # NDCG = []

                        for rank_ in range(1, len(pred_documents)):
                            rank_value = rank_ + 1
                            if pred_documents[
                                    rank_] in score_relations[query_id]:
                                DCG.append(
                                    score_relations[query_id][
                                        pred_documents[rank_]
                                        ]/math.log2(rank_value))
                            else:
                                DCG.append(0)
                            DCG_summation.append(DCG_summation[-1]+DCG[-1])

                            if len(perfect_sort) - 1 >= rank_:
                                PERFECT_DCG.append(
                                    perfect_sort[
                                        rank_][1] / math.log2(rank_value))
                            else:
                                PERFECT_DCG.append(0)

                            PERFECT_summation.append(
                                PERFECT_summation[-1]+PERFECT_DCG[-1])

                        ndcg = []
                        for k in range(len(DCG_summation)):
                            if -1e-12 < PERFECT_summation[k] < 1e-9:
                                ndcg.append(0)
                            else:
                                ndcg.append(
                                    DCG_summation[k] / PERFECT_summation[k])

                        for i in range(len(ndcg)):
                            total_dcg[i] += DCG_summation[i]
                            total_ndcg[i] += ndcg[i]

                    # Hit calculations
                    for p in pred_documents:
                        if p in self.relevance_dictionary[query_id]:
                            total_hit[r] += 1
                            break
                    pred_documents = set(pred_documents)
                    precision_score = nltk.scores.precision(
                        reference, pred_documents)
                    recall_score = nltk.scores.recall(
                        reference, pred_documents)
                    f1_score = nltk.scores.f_measure(
                        reference, pred_documents, 1.0)

                    total_precision[r] += precision_score
                    total_recall[r] += recall_score
                    total_f1[r] += f1_score

        # total_prediction_count is the number of queries...

        for r in scores_at:
            total_hit[r] = total_hit[r] / total_prediction_count * 100
            total_precision[r] /= total_prediction_count
            total_recall[r] /= total_prediction_count
            total_f1[r] /= total_prediction_count
            total_map[r] /= total_prediction_count
            total_mrr[r] /= total_prediction_count

        for i in range(len(calculated_precisions_at_11)):
            calculated_precisions_at_11[i] /= total_prediction_count

        for i in range(len(total_dcg)):
            total_dcg[i] /= total_prediction_count
            total_ndcg[i] /= total_prediction_count

        for r in total_hit:
            print("Hit @%d : %.2f" % (r, total_hit[r]))

        for r in total_precision:
            print("Average Precision @%d : %.2f" % (r, total_precision[r]))

        for r in total_recall:
            print("Average Recall @%d : %.2f" % (r, total_recall[r]))

        for r in total_f1:
            print("Average F1 @%d : %.2f" % (r, total_f1[r]))

        for r in total_map:
            print("Average MAP @%d : %.2f" % (r, total_map[r]))

        for r in total_mrr:
            print("Average MRR @%d : %.2f" % (r, total_mrr[r]))

        print("Averaged 11 point precision values ")
        print(calculated_precisions_at_11)

        print("Average DCG")
        print(total_dcg)
        for a in scores_at:
            a = total_dcg[a-1]
            print("%.3f" % a, end=" ")

        print("Average NDCG")
        print(total_ndcg)
        for a in scores_at:
            a = total_ndcg[a-1]
            print("%.3f" % a, end=" ")

        print("Query count : ", total_prediction_count)
        plt.plot(
            [i / 10 for i in range(0, 11)],
            calculated_precisions_at_11, 'g-o', label="BM25")

        plt.legend(loc="upper right")
        plt.xlabel("Recall")
        plt.ylabel("Average Precision")
        plt.show()

        if self.ranking_method.method_name == "random":
            result_path = (
                "Results/cord19(randomranking)"+str(time.time()*1000)+".pkl")
        else:
            result_path = (
                "Results/cord19("+self.ranking_method.method_name+").pkl")

        pickle.dump((
            total_hit,
            total_precision,
            total_recall,
            total_f1,
            total_map,
            total_mrr,
            calculated_precisions_at_11,
            total_dcg,
            total_ndcg
            ), open(result_path, "wb"))
