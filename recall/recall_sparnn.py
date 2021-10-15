import config
from recall import pysparnn_embedding
from typing import List, Text, Dict, Any
import numpy as np


class RecallBySparrn:
    def __init__(self, method, by_word=False):
        """
        :param method:  tfidf  bm25  fasttext
        :param by_word:
        """
        self.by_word = by_word
        sent_vec = pysparnn_embedding.PySparnnEmbedding(by_word=by_word, method=method)
        self.qa_dict = sent_vec.qa_dict
        self.vectorizer, self.features_vec, self.q_cuted, self.sparnn_index = sent_vec.build_vector()

    def predict(self, sentence: List[Text], return_distance=True):
        # 将待预测的句子生成句子向量
        cur_sent_vec = self.vectorizer.transform(sentence)
        # 根据句子向量从向量矩阵中查找对于最相近的句子
        ret = self.sparnn_index.search(
            cur_sent_vec,
            k=config.sparrn_topk,
            k_clusters=config.sparrn_clusters,
            return_distance=return_distance
            )

        return ret


if __name__ == '__main__':
    recall = RecallBySparrn(method='fasttext')

    sentence = ['Python 常见 数据结构 有 哪些']
    ret = recall.predict(sentence)
    print(ret)
