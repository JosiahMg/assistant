"""
小黄鸡语料库构建词典
"""
from typing import Text, List
import pickle
import torch


class Vocab:
    """此项目无需词典
    """
    PAD_TAG = 'PAD'
    UNK_TAG = 'UNK'
    SOS_TAG = 'SOS'
    EOS_TAG = 'EOS'
    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3

    def __init__(self):
        self.dict = {self.PAD_TAG: self.PAD,
                     self.UNK_TAG: self.UNK,
                     self.SOS_TAG: self.SOS,
                     self.EOS_TAG: self.EOS
                     }
        self.count = {}

    def fit(self, sencence):
        """

        :param sencence: 句子
        :return:
        """
        for word in sencence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min_count=3, max_count=None, max_feature=None):
        """
        构造词典
        :param min_count:
        :param max_count:
        :param max_feature:
        :return:
        """
        temp = self.count.copy()
        for key in temp.keys():
            cur_count = self.count.get(key, 0)
            if min_count and cur_count < min_count:
                del self.count[key]
            if max_count and cur_count > max_count:
                del self.count[key]
        if max_feature:
            self.count = dict(
                sorted(self.count.items(), key=lambda x: x[1], reverse=True)[:max_feature])
        for key in self.count:
            self.dict[key] = len(self.dict)

        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence: List[Text], max_len, add_eos=False):
        if len(sentence) > max_len:
            sentence = sentence[:max_len]
        seq_len = len(sentence)
        if add_eos:
            sentence = sentence + [self.EOS_TAG]
        if seq_len < max_len:
            sentence = sentence + [self.PAD_TAG]*(max_len-seq_len)
        result = [self.dict.get(i, self.UNK) for i in sentence]
        return result

    def inverse_transform(self, indices):
        # return [self.inverse_dict.get(i, self.UNK_TAG) for i in indices]
        result = []
        for i in indices:
            if i == self.EOS or i == self.PAD:
                break
            result.append(self.inverse_dict.get(i, self.UNK_TAG))
        return result

    def __len__(self):
        return len(self.dict)

    def save_vocab(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    @staticmethod
    def load_vocab(filename):
        return pickle.load(open(filename, 'rb'))
