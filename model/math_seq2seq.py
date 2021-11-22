import re
import torch
import torch.nn as nn
import config
from vocabs.vocab import Vocab
from model.base_lib.seq2seq import Seq2seq
from tqdm import tqdm
from datasets import math_dataset
import torch.nn.functional as F
from tokenizers.space_tokenizer import SpaceTokenizer
import os
from pprint import pprint
from utils.utils import is_equal
from log.log import make_log

logger = make_log()
que_vocab = Vocab().load_vocab(config.math_vocab_question_path)
equ_vocab = Vocab().load_vocab(config.math_vocab_equation_path)



class CorrectEquation:
    """
    校正 Equation 表达式，使得可以通过eval函数进行计算
    """

    def remove_bucket(self, equation):
        """去掉冗余的括号
        """
        l_buckets, buckets = [], []
        for i, c in enumerate(equation):
            if c == '(':
                l_buckets.append(i)
            elif c == ')':
                buckets.append((l_buckets.pop(), i))
        eval_equation = eval(equation)
        for l, r in buckets:
            new_equation = '%s %s %s' % (
                equation[:l], equation[l + 1:r], equation[r + 1:]
            )
            try:
                if self.is_equal(eval(new_equation.replace(' ', '')), eval_equation):
                    equation = new_equation
            except:
                pass
        return equation.replace(' ', '')

    def __call__(self, equation):
        equation = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', equation)
        equation = re.sub('(\d+)\(', '\\1+(', equation)
        # 处理百分数
        equation = re.sub('([\.\d]+)%', '(\\1/100)', equation)
        # 冒号转除号、剩余百分号处理
        equation = equation.replace(':', '/').replace('%', '/100')
        if equation[:2] == 'x=':
            equation = equation[2:]
        equation = self.remove_bucket(equation)
        return equation


class MathExtractEquationByRegex:
    """
    基于正则表达式找到公式
    1. 找到equation
    2. 纠正equation: CorrectEquation()()
    return: equation
    """
    def __call__(self, equation):
        return equation


que_vocab = Vocab().load_vocab(config.math_vocab_question_path)
equ_vocab = Vocab().load_vocab(config.math_vocab_equation_path)


seq2seq = Seq2seq(que_vocab, equ_vocab, config.math_emb_size, config.math_num_layer, config.math_hidden_size,
                  config.math_batch_first, config.math_bidirection, config.teach_forcing_rate,
                  config.math_max_equa_len).to(config.device)

optimizer = torch.optim.Adam(seq2seq.parameters(), lr=0.001)


# 训练模型
def train(epoch):
    """
    input.shape: (batch, seq_len)
    target.shape: (batch, seq_len)
    input_len: (batch,)
    target_len: (batch,)
    """
    # 加载之前保存的模型进行训练
    if not config.math_retrain and os.path.exists(config.math_model):
        checkpoint = torch.load(config.math_model, map_location=config.device)
        seq2seq.load_state_dict(checkpoint)

    seq2seq.train()

    bar = tqdm(enumerate(math_dataset.trian_dataloader), total=len(math_dataset.trian_dataloader),
               ascii=True, desc="EP_train:%d" % (epoch), bar_format="{l_bar}{r_bar}")

    for index, (inputs, target, _, input_len, target_len) in bar:

        inputs = inputs.to(config.device)
        target = target.to(config.device)
        # input_len = input_len
        # target_len = target_len

        optimizer.zero_grad()
        # decoder_outputs.shape: (batch, seq_len, vocab_size)

        decoder_outputs, _ = seq2seq(inputs, target, input_len, target_len)
        decoder_outputs = decoder_outputs.view(-1, len(equ_vocab))  # (batch*seq, vocab_size)
        target = target.view(-1)  # (batch*seq,)

        loss = F.nll_loss(decoder_outputs, target, ignore_index=equ_vocab.PAD)
        loss.backward()
        optimizer.step()

        if index % 100 == 0:
            bar.set_description('epoch: {}\tidx:{}\tloss:{:.4f}'.format(
                epoch, index, loss.item()))
            logger.info('epoch: {}\tidx:{}\tloss:{:.4f}'.format(
                epoch, index, loss.item()))
            torch.save(seq2seq.state_dict(), config.math_model)
            torch.save(optimizer.state_dict(), config.math_opt)


# 评估模型， 每个epoch结束后执行
def evaluate():
    seq2seq.eval()
    total_count = 0
    correct_count = 0
    acc = 0.0
    bar = tqdm(enumerate(math_dataset.valid_dataloader), total=len(math_dataset.test_dataloader),
               ascii=True, desc="EP_valid ", bar_format="{l_bar}{r_bar}")
    # 1. 准备数据
    for index, (inputs, target, ans, input_len, target_len) in bar:
        inputs = inputs.to(config.device)
        # target = target
        # input_len = input_len
        # target_len = target_len
        total_count += inputs.size(0)
        indices = seq2seq.evaluate(inputs, input_len)  # (1, seq_len)
        for i, elem in enumerate(indices):
            outputs = equ_vocab.inverse_transform(elem)
            outputs = ''.join(outputs)
            try:
                if is_equal(eval(outputs), ans[i]):
                    correct_count += 1
            except:
                pass
    acc = correct_count / total_count
    logger.info(f'Eval accuarcy: {acc}')
    return acc


def predict():
    # 1. 准备数据
    inputs = input('请输入:')
    inputs = [SpaceTokenizer().tokenize(inputs)]

    input_lens = torch.LongTensor(
        [len(i) if len(i) < config.math_max_ques_len else config.math_max_ques_len for i in inputs])
    inputs = torch.LongTensor([que_vocab.transform(i, max_len=config.math_max_ques_len) for i in inputs]).to(config.device)

    # 2. 加载模型
    seq2seq = Seq2seq(que_vocab, equ_vocab, config.math_emb_size, config.math_num_layer, config.math_hidden_size,
                      config.math_batch_first, config.math_bidirection, config.teach_forcing_rate,
                      config.math_max_equa_len).to(config.device)
    seq2seq.load_state_dict(torch.load(config.math_model))

    # 3. 模型预测
    indices = seq2seq.evaluate(inputs, input_lens)  # (1, seq_len)

    # 4. 反序列化
    for i, elem in enumerate(indices):
        outputs = equ_vocab.inverse_transform(elem)
        outputs = ''.join(outputs)
        pprint(outputs)
        try:
            print(eval(outputs))
        except Exception as e:
            print(e.args)


if __name__ == '__main__':
    if torch.cuda.is_available():
        logger.info('We will use the GPU:%s', torch.cuda.get_device_name(0))
    else:
        logger.info('No GPU available, using the CPU instead.')

    train_mode = True
    if train_mode:
        for epoch in range(100):
            train(epoch)
            evaluate()
    else:
            predict()
