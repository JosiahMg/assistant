from pre_corpus import pre_origin_corpus
from recall import recall_sparnn
from recall import recall_inverted_index
from pprint import pprint

if __name__ == '__main__':
    # 预处理数据
    pre_origin_corpus.pre_origin_corpus()


    # 使用pysparnn进行召回
    recall = recall_sparnn.RecallBySparnn(method='fasttext')

    sentence = ['Python 常见 数据结构 有 哪些']
    ret = recall.predict(sentence, return_distance=False)
    pprint(ret)


    # 使用inverted index进行召回
    recall = recall_inverted_index.RecallByInvertedIndex()
    recall_qlist, recall_alist = recall.filter_question_invert_tab(['python 常见 数据结构 有 哪些'])
    pprint(recall_qlist)




