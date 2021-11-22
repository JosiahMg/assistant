import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from vocabs.vocab import Vocab
import config
from datasets import xhj_dataset
from model.base_lib.seq2seq import Seq2seq
from tokenizers.space_tokenizer import SpaceTokenizer
"""
训练过程:
1. 实例化model optimizer loss
2. 遍历dataloader
3. 计算output
4. 计算loss
5. 模型保存和加载
"""
emb_size = config.xhj_emb_size
num_layer = config.xhj_num_layer
hidden_size = config.xhj_hidden_size
batch_first = config.xhj_batch_first
bidirection = config.xhj_bidirection
device = config.device
xhj_vocab = Vocab().load_vocab(config.xhj_vocab_path)

seq2seq = Seq2seq(xhj_vocab, xhj_vocab, emb_size, num_layer, hidden_size, batch_first,
                  bidirection, config.teach_forcing_rate, config.xhj_max_decoder_len).to(device)

optimizer = torch.optim.Adam(seq2seq.parameters(), lr=0.001)


def train(epoch):
    """
    input.shape: (batch, seq_len)
    target.shape: (batch, seq_len)
    input_len: (batch,)
    target_len: (batch,)
    """
    # 加载之前保存的模型进行训练
    if not config.xhj_retrain:
        checkpoint = torch.load(config.xhj_model, map_location=device)
        seq2seq.load_state_dict(checkpoint)

    bar = tqdm(enumerate(xhj_dataset.train_data_loader), total=len(
        xhj_dataset.train_data_loader), ascii=True, desc='train')
    for index, (inputs, target, input_len, target_len) in bar:

        inputs, target, input_len, target_len = inputs.to(device), target.to(
            device), input_len.to(device), target_len.to(device)

        optimizer.zero_grad()
        # decoder_outputs.shape: (batch, seq_len, vocab_size)
        decoder_outputs, _ = seq2seq(inputs, target, input_len, target_len)

        decoder_outputs = decoder_outputs.view(decoder_outputs.size(
            0)*decoder_outputs.size(1), -1)  # (batch*seq, vocab_size)
        target = target.view(-1)  # (batch*seq,)

        loss = F.nll_loss(decoder_outputs, target, ignore_index=xhj_vocab.PAD)
        loss.backward()
        optimizer.step()

        if index % 100 == 0:
            bar.set_description('epoch: {}\tidx:{}\tloss:{:.4f}'.format(
                epoch, index, loss.item()))
            torch.save(seq2seq.state_dict(), config.xhj_model)
            torch.save(optimizer.state_dict(), config.xhj_opt)


def eval():
    # 1. 准备数据
    inputs = input('请输入:')
    inputs = [SpaceTokenizer().tokenize(inputs)]
    inputs = torch.LongTensor([xhj_vocab.transform(i, max_len=config.xhj_max_encoder_len) for i in inputs]).to(device)
    input_lens = torch.LongTensor([len(inputs) if len(inputs)<config.xhj_max_encoder_len else config.xhj_max_encoder_len]).to(device)

    # 2. 加载模型
    seq2seq = Seq2seq(xhj_vocab, xhj_vocab, emb_size, num_layer, hidden_size, batch_first,
                      bidirection, config.teach_forcing_rate, config.xhj_max_decoder_len).to(device)
    seq2seq.load_state_dict(torch.load(config.xhj_model))

    # 3. 模型预测
    indices = seq2seq.evaluate(inputs, input_lens)  # (1, seq_len)

    # 4. 反序列化
    for elem in indices:
        outputs = xhj_vocab.inverse_transform(elem)
        print('answer: ', ''.join(outputs))


if __name__ == '__main__':
    train_mode = False
    if train_mode:
        for epoch in range(100):
            train(epoch)
    else:
        while True:
            eval()


