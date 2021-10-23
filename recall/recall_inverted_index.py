import config
import pandas as pd
import json
from typing import List, Text, Tuple, Dict, Set, Union, Any
from pprint import pprint
from collections import Counter
import os


class RecallByInvertedIndex:
    def __init__(self, retrain=False):
        self.retrain = retrain
        self.qa_list = pd.read_csv(
            config.merge_qa_csv_path, encoding='utf-8', header=[0])
        # drop na的索引
        self.qa_list.dropna(axis=0, how='any', inplace=True)
        # 重新排序
        self.qa_list.reset_index(drop=True, inplace=True)
        self.inverted_index_json_path = config.inverted_index_path
        self.inverted_index = self.__inverted_index()

    def __inverted_index(self):
        """
        docid: 问答索引
        tf: keyword在一个doc中出现的次数  counter(token) in each document
        position: keyword在doc中的偏移
        idf: keyword的idf值  log_e(doc_n/counter(token))

        :return: 倒排索引
        format:
        {
            "keyword": {
                "docs": [  # 倒排表
                    {"docid": int, "tf": int, "position": Tuple[int, int]},
                    {"docid": int, "tf": int, "position": Tuple[int, int]}, ...
                ],
                "idf": float
            },
            ...
        }
        """
        #TODO 增加idf值 如果召回的数量过多 可以通过idf值进行过滤

        if os.path.exists(self.inverted_index_json_path) and not self.retrain:
            return json.load(open(self.inverted_index_json_path, encoding='utf-8'))

        invert_index = {}  # 倒排索引
        idf_count = Counter()

        for idx, kws in enumerate(self.qa_list['question']):
            kws_list = kws.split()
            # 计算每个document的词频
            tf_count = Counter(kws_list)
            # 计算每个idf值
            for kw in set(kws_list):
                idf_count[kw] += 1

            for kw in kws_list:
                invert_tab = {"docid": idx, "tf": tf_count[kw], "position": [0, 0]}
                if kw in invert_index.keys():
                    invert_index[kw]["docs"].append(invert_tab)
                    invert_index[kw]["idf"] = idf_count[kw]
                else:
                    invert_index[kw] = {"docs": [invert_tab], 'idf': idf_count[kw]}  # kw的倒排表

        json.dump(invert_index, open(self.inverted_index_json_path, mode='w',
                                encoding='utf-8'), ensure_ascii=False, indent=2)
        return invert_index

    # Text: document
    def filter_question_invert_tab(self, split_questions: List[Text]):
        idx_set = set()
        questions = []
        answers = []
        for doc in split_questions:
            question = []
            answer = []
            for kw in doc.split():
                if kw in self.inverted_index.keys():
                    for invert_tab_dict in self.inverted_index[kw]['docs']:
                        idx_set.add(invert_tab_dict['docid'])
            for idx in idx_set:
                question.append(self.qa_list['question'][idx])
                answer.append(self.qa_list['answer'][idx])
            questions.append(question)
            answers.append(answer)
        return questions, answers


if __name__ == '__main__':
    recall = RecallByInvertedIndex()
    recall_qlist, recall_alist = recall.filter_question_invert_tab(['python 常见 数据结构 有 哪些'])
    pprint(recall_qlist)
    # pprint(recall_alist)


