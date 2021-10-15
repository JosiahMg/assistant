import numpy as np
import fasttext
import fasttext.util

# download an english model
fasttext.util.download_model('en', if_exists='ignore')  # English
model = fasttext.load_model('cc.en.300.bin')

# Getting word vectors for 'one' and 'two'.
embedding_1 = model.get_sentence_vector('baby dog')
embedding_2 = model.get_word_vector('puppy')
embedding_3 = model.get_sentence_vector('puppy')

print(embedding_1)
print(embedding_2)
print(embedding_3 == embedding_2)