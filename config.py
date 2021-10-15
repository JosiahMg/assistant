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
ft_embedding_path = os.path.join(root_dir, 'model/fasttext_embedding')

# the model of word or phrase embedding by pysparrn
sparrn_embedding_path = os.path.join(root_dir, 'model/sparrn_embedding')

# fasttext train_unsupervised hyper-parameter
wordNgrams = 1
epoch = 20
minCount = 2

# sparrn hyper-parameter
sparrn_topk = 10
sparrn_clusters = 10
