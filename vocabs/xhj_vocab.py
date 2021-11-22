import config
from vocabs.vocab import Vocab

if __name__ == '__main__':
    vocab = Vocab()
    for line in open(config.mid_xiaohuangji_question_path, encoding='utf-8').readlines():
        vocab.fit(line.strip().split())
    for line in open(config.mid_xiaohuangji_answer_path, encoding='utf-8').readlines():
        vocab.fit(line.strip().split())
    vocab.build_vocab()
    print(len(vocab))
    vocab.save_vocab(config.xhj_vocab_path)

    xhj_vocab = Vocab().load_vocab(config.xhj_vocab_path)
    print(xhj_vocab.dict)

