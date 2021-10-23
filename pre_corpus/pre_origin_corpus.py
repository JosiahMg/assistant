import config
import os
import pandas as pd
import jieba
import json
from tokenizers import jieba_tokenizer
import time

ignore_corpus = ['xiaohuangji']


class PreOriginCorpus:
    def __init__(self, tokenizer=None):
        """
        处理原始数据集
        :param input_dir: The directory of origin corpus
        :param output_path: The path of output
        """
        # TODO 原始数据集使用rasa格式的数据
        self.inputs = config.origin_corpus_dir
        self.qa_csv_path = config.merge_qa_csv_path
        self.q_csv_path = config.split_q_path
        self.a_csv_path = config.split_a_path
        self.qa_json_path = config.merge_qa_json_path
        # TODO 同义词替换 纠错 标点符号替换
        self.tokenizer = tokenizer

    def merge_qa(self):
        """
        merge origin corpus to merge_qa.csv,
        split question and answer to split_a.csv and split_csv.csv, meanwhile you can token by tokenizer
        tokenizer: default by jieba

        merge origin corpus to merge_qa.json
        format:
        {
            "what is python": {
                                "answer": "python is ...",
                                "q_cut": ['what', 'is', 'python'],
                                "entity": ['python']
                               },
            ...
        }
        """

        def token_strip(token):
            return token.strip(' ')

        def tokenizer_join_str(x):
            if self.tokenizer:
                return ' '.join(list(map(token_strip, self.tokenizer.tokenize(x))))
            else:
                return x

        df_merge = []
        qa_dict = {}

        for root, dirs, files in os.walk(self.inputs):
            label = os.path.basename(root)
            if label in ignore_corpus:
                print(f'Ignore fold {label}')
                continue
            for file in files:
                file_path = os.path.join(root, file)
                print(f'Now processing file {file_path}')
                # process for  csv
                df = pd.read_csv(file_path, encoding='utf-8',
                                 sep=',', header=[0])
                # drop含有nan的行，axis=0: 按行删除
                df.dropna(axis=0, how='any', inplace=True)
                # 需要重新排序
                df.reset_index(drop=True, inplace=True)
                df['label'] = label
                df_merge.append(df)

                # process for  json
                for q, a in zip(df['question'], df['answer']):
                    tokens = tokenizer_join_str(q)
                    qa_dict[tokens] = {}
                    qa_dict[tokens]['answer'] = a.strip()
                    qa_dict[tokens]['q_cut'] = self.tokenizer.tokenize(q) if self.tokenizer else []
                    # TODO 样本中需要添加实体并在代码中解析出来
                    qa_dict[tokens]['entity'] = []
                    # 当前QA问答对属于label类型
                    qa_dict[tokens]['label'] = label

        # save csv
        df_merge = pd.concat(df_merge)
        df_merge['question'] = df_merge['question'].apply(tokenizer_join_str)
        df_merge['question'].to_csv(self.q_csv_path, index=False, header=None)
        df_merge['answer'].to_csv(self.a_csv_path, index=False, header=None)
        df_merge.to_csv(self.qa_csv_path, index=False, header=[
                        'question', 'answer', 'label'])

        # save json
        json.dump(qa_dict, open(self.qa_json_path, mode='w',
                                encoding='utf-8'), ensure_ascii=False, indent=2)


def pre_origin_corpus():
    tokenizer = jieba_tokenizer.JiebaTokenizer()
    pr = PreOriginCorpus(tokenizer)
    pr.merge_qa()


if __name__ == '__main__':
    pre_origin_corpus()
