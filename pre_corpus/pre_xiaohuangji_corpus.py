"""
准备小黄鸡语料, 用于搭建闲聊对话系统

"""
from vocabs.vocab import Vocab
import config
import string
from tokenizers.space_tokenizer import SpaceTokenizer
from tqdm import tqdm


def filter(pair):
    """
    :param pair: [question, answer]
    :return:
    """
    if pair[0] in list(string.ascii_lowercase):
        return True
    if pair[1].count('=') > 2:
        return True
    if len(pair[0][0].strip()) == 0 or len(pair[1][0].strip()) == 0:
        return True
    return False


def prepar_xiaohuangji(by_word=False):
    path = config.origin_xiaohuangji_path
    input_path = config.mid_xiaohuangji_question_path
    target_path = config.mid_xiaohuangji_answer_path
    f_input = open(input_path, encoding='utf-8', mode='w')
    f_target = open(target_path, encoding='utf-8', mode='w')
    one_qa_pair = []
    num = 0
    for line in tqdm(open(path, encoding='utf-8').readlines(), ascii=True, desc='小黄鸡语料处理进度'):
        if num == 10000:
            break
        if line.startswith('E'):
            continue
        else:
            line = line[1:].strip().lower()
            if by_word:  # 安装word进行分词
                line = SpaceTokenizer().tokenize(line)
                line = " ".join(line) + '\n'
            if len(one_qa_pair) < 2:
                one_qa_pair.append(line)
            if len(one_qa_pair) == 2:
                # 判断句子是否合规
                if filter(one_qa_pair):
                    one_qa_pair = []
                    continue
                f_input.write(one_qa_pair[0])
                f_target.write(one_qa_pair[1])
                num += 1
                one_qa_pair = []

    f_input.close()
    f_target.close()
    return num


if __name__ == '__main__':
    print(prepar_xiaohuangji(by_word=True))