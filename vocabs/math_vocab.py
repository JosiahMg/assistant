import config
from vocabs.vocab import Vocab
import pandas as pd
from pprint import pprint


if __name__ == '__main__':
    vocab_question = Vocab()
    vocab_equation = Vocab()
    math_df = pd.read_csv(config.mid_math_ape_train_path, encoding='utf-8', header=[0], sep=',')
    for q, e in math_df.loc[:, ['question', 'equation']].values:
        q = q.strip()
        e = e.strip()
        vocab_question.fit(q)
        vocab_equation.fit(e)

    vocab_question.build_vocab()
    vocab_equation.build_vocab(min_count=0)

    vocab_question.save_vocab(config.math_vocab_question_path)
    vocab_equation.save_vocab(config.math_vocab_equation_path)

    print(len(vocab_question))
    print(len(vocab_equation))

    print(vocab_question.dict)
    print(vocab_equation.dict)
