"""
使用pysparrn库 facebook
将训练集question以及对于的sentence vector构造成一一对应的关系，即matrix
对于给定的新的句子的vector传入 matrix 通过类似kmeans方法获取训练集中最相似的topN个结果
"""


import config
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from recall import fasttext_vectorizer
from recall import bm25_vectorizer
import pysparnn.cluster_index as ci
import os
import pickle


class PySparrnEmbedding:
    """
    使用pysparrn构造embedding
    """
    def __init__(self, by_word=False, method='tfidf'):

        self.sparrn_train_data_path = config.merge_qa_json_path  # 训练需要的数据
        self.sparrn_model_path = config.sparrn_embedding_path  # 存储sparrn模型的路径
        self.ft_train_data_path = config.split_q_path   # 训练fasttext需要分割后的question数据
        self.ft_emb_path = config.ft_embedding_path     # 存储fasttext模型的路径
        self.by_word = by_word
        self.method = method
        self.qa_dict = json.load(open(self.sparrn_train_data_path, 'r', encoding='utf-8'))
        self.sparrn_model_path = self.sparrn_model_path + f'.{method}'
        if method.lower() == 'bm25':
            self.vectorizer = bm25_vectorizer.Bm25Vectorizer()
        elif method.lower() == 'fasttext':
            self.vectorizer = fasttext_vectorizer.FastTextVectorizer(by_word, retrain=False)
        elif method.lower() == 'tfidf':
            if self.by_word:
                self.vectorizer = TfidfVectorizer(analyzer='char', lowercase=False)
            else:
                self.vectorizer = TfidfVectorizer()
        else:
            raise NotImplemented

    def build_vector(self):
        q_cuted = [qa for qa in self.qa_dict.keys()]
        print(f'Begin {self.method} vector ...')
        feature_vec = self.vectorizer.fit_transform(q_cuted)
        sparnn_index = self.__get_cp(feature_vec, q_cuted)
        return self.vectorizer, feature_vec, q_cuted, sparnn_index

    def __get_cp(self, vectors, data):
        if os.path.exists(self.sparrn_model_path):
            print('Begin load pysparrn embedding ...')
            sparnn_index = pickle.load(open(self.sparrn_model_path, 'rb'))
        else:
            print('Begin create pysparrn vector ...')
            sparnn_index = self.__build_cp(vectors, data)
        return sparnn_index

    def __build_cp(self, vectors, data):
        sparnn_index = ci.MultiClusterIndex(vectors, data)
        pickle.dump(sparnn_index, open(self.sparrn_model_path, 'wb'))
        return sparnn_index


if __name__ == '__main__':
    sv = PySparrnEmbedding(by_word=False, method='fasttext')
    print(sv.build_vector())
