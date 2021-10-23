"""
使用fasttext的构建句子向量
"""
import config
import fasttext
import os
import numpy as np
from typing import List, Text, Dict, Tuple


class FastTextVectorizer:
    def __init__(self, max_features=100, by_word=False, retrain=False):
        self.train_data_path = config.split_q_path
        self.save_model_path = config.ft_embedding_path + \
            ('_word' if by_word else '')
        self.by_word = by_word
        self.retrain = retrain
        self.wordNgrams = config.wordNgrams
        self.epoch = config.epoch
        self.minCount = config.minCount
        self.model = self.fit(max_features)

    def fit(self, max_features):
        if os.path.exists(self.save_model_path) and not self.retrain:
            self.model = fasttext.load_model(self.save_model_path)
        else:
            self.model = fasttext.train_unsupervised(self.train_data_path,
                                                     wordNgrams=self.wordNgrams,
                                                     epoch=self.epoch,
                                                     minCount=self.minCount,
                                                     dim=max_features)
            self.model.save_model(self.save_model_path)
        return self.model

    def transform(self, sentences: List[Text]):
        results = [self.model.get_sentence_vector(q) for q in sentences]
        return np.array(results)

    def fit_transform(self, sentences: List[Text]):
        return self.transform(sentences)


if __name__ == '__main__':
    ft = FastTextVectorizer()
    st = ['Python 是 什么']
    vec = ft.fit_transform(st)
    print(vec.shape)
