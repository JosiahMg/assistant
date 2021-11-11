import os

# The root directory of the project
root_dir = os.path.dirname(__file__)

# The directory of save model
model_dir = os.path.join(root_dir, 'model')

# The directory of original corpus
origin_corpus_dir = os.path.join(root_dir, 'corpus/origin')

# merge all csv file to one csv
merge_qa_csv_path = os.path.join(root_dir, 'corpus/merge_data/merge_qa.csv')

# split qa to q.csv for trainning fasttext.train_unsupervised()
split_q_path = os.path.join(root_dir, 'corpus/merge_data/split_q.csv')

# split qa to a.csv
split_a_path = os.path.join(root_dir, 'corpus/merge_data/split_a.csv')

# merge qa to json for trainning pysparrn model
merge_qa_json_path = os.path.join(root_dir, 'corpus/merge_data/merge_qa.json')


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

inverted_index_path = os.path.join(root_dir, 'corpus/merge_data/inverted_index.json')


# log.conf
log_conf = os.path.join(root_dir, 'log/log.conf')

# math ape210k data
json_math_test_path = os.path.join(origin_corpus_dir, 'math/ape210k/test.ape.json')
json_math_train_path = os.path.join(origin_corpus_dir, 'math/ape210k/train.ape.json')
json_math_valid_path = os.path.join(origin_corpus_dir, 'math/ape210k/valid.ape.json')

# math origin created by ape210k data
math_ape_path = os.path.join(origin_corpus_dir, 'math/math_ape.csv')
