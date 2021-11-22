import os
import torch
import pickle
from vocabs.vocab import Vocab

######## global parameters #########
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The root directory of the project
root_dir = os.path.dirname(__file__)

# The directory of save model
model_dir = os.path.join(root_dir, 'model')

# The directory of original corpus
origin_corpus_dir = os.path.join(root_dir, 'corpus/origin')
mid_corpus_dir = os.path.join(root_dir, 'corpus/middle')
final_corpus_dir = os.path.join(root_dir, 'corpus/final')

warehouse_path = os.path.join(root_dir, 'warehouse')

# merge all csv file to one csv
merge_qa_csv_path = os.path.join(final_corpus_dir, 'merge_qa.csv')

# split qa to q.csv for trainning fasttext.train_unsupervised()
split_q_path = os.path.join(final_corpus_dir, 'split_q.csv')

# split qa to a.csv
split_a_path = os.path.join(final_corpus_dir, 'split_a.csv')

# merge qa to json for trainning pysparrn model
merge_qa_json_path = os.path.join(final_corpus_dir, 'merge_qa.json')


# the model of word or phrase embedding by fasttext
ft_embedding_path = os.path.join(root_dir, 'warehouse/fasttext_embedding')

# the model of word or phrase embedding by pysparrn
sparrn_embedding_path = os.path.join(root_dir, 'warehouse/sparrn_embedding')

# fasttext train_unsupervised hyper-parameter
wordNgrams = 3
epoch = 10
minCount = 5

# sparrn hyper-parameter
sparrn_topk = 20
sparrn_clusters = 10


# stopwords
stopword_path = os.path.join(root_dir,'tokenizers/stopwords/stopword.txt')
# userdicts
userdict_path = os.path.join(root_dir,'tokenizers/userdicts/userdict.txt')

inverted_index_path = os.path.join(final_corpus_dir, 'inverted_index.json')


################# log #################
log_conf = os.path.join(root_dir, 'log/log.conf')
log_file = os.path.join(root_dir, 'log/assitant.log')


################# 小黄鸡 #################
# xiaohuangji corpus
origin_xiaohuangji_path = os.path.join(origin_corpus_dir, 'xiaohuangji/xiaohuangji.conv')
mid_xiaohuangji_answer_path = os.path.join(mid_corpus_dir, 'xiaohuangji/xiaohuangji_answer.txt')
mid_xiaohuangji_question_path = os.path.join(mid_corpus_dir, 'xiaohuangji/xiaohuangji_question.txt')

# 保存词典
xhj_vocab_path = os.path.join(warehouse_path, 'xhj.vocab')
xhj_max_encoder_len = 20  # 30: 90%
xhj_max_decoder_len = 20    # 30: 90%
xhj_batch_size = 256
teach_forcing_rate = 0.3  # teacher forcing rate

xhj_emb_size = 128
xhj_num_layer = 3
xhj_hidden_size = 256
xhj_batch_first = True
xhj_bidirection = True

xhj_model = os.path.join(warehouse_path, 'xhj_seq2seq.model')
xhj_opt = os.path.join(warehouse_path, 'xhj_adam.opt')

xhj_retrain = True
beam_search = False
xhj_beam_width = 3

################# math ape210k data #################

origin_math_ape_train_path = os.path.join(origin_corpus_dir, 'math/ape210k/train.ape.json')
origin_math_ape_test_path = os.path.join(origin_corpus_dir, 'math/ape210k/test.ape.json')
origin_math_ape_valid_path = os.path.join(origin_corpus_dir, 'math/ape210k/valid.ape.json')

# math origin created by ape210k data
mid_math_ape_train_path = os.path.join(mid_corpus_dir, 'math/train_ape.csv')
mid_math_ape_test_path = os.path.join(mid_corpus_dir, 'math/test_ape.csv')
mid_math_ape_valid_path = os.path.join(mid_corpus_dir, 'math/valid_ape.csv')
mid_math_oper_train_path = os.path.join(mid_corpus_dir, 'math/operation_train.csv')
mid_math_oper_test_path = os.path.join(mid_corpus_dir, 'math/operation_test.csv')

# vocab
math_vocab_question_path = os.path.join(warehouse_path, 'math_question.vocab')
math_vocab_equation_path = os.path.join(warehouse_path, 'math_equation.vocab')

math_max_ques_len = 100
math_max_equa_len = 80
math_batch_size = 32
math_emb_size = 128
math_num_layer = 2
math_hidden_size = 512
math_batch_first = True
math_bidirection = True

math_retrain = False
math_beam_search = False
math_beam_width = 3

math_model = os.path.join(warehouse_path, 'math_seq2seq.model')
math_opt = os.path.join(warehouse_path, 'math_adam.opt')
