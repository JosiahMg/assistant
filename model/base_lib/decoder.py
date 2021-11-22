import torch
import torch.nn as nn
from vocabs.vocab import Vocab
import config
import random
import torch.nn.functional as F
from model.base_lib import attention, beam_search
import numpy as np


class Decoder(nn.Module):
    def __init__(self, vocab, emb_size, num_layer, hidden_size, batch_first=True,
                 bidirection=False, teach_forcing_rate=0.7, max_seq_len=30):
        super(Decoder, self).__init__()
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.teach_forcing_rate = teach_forcing_rate  # teach forcing机制
        self.embedding = nn.Embedding(
            num_embeddings=len(vocab), embedding_dim=emb_size, padding_idx=vocab.PAD)

        self.gru = torch.nn.GRU(input_size=emb_size, num_layers=num_layer,
                                hidden_size=hidden_size, batch_first=batch_first, bidirectional=bidirection)
        self.fc = nn.Linear(hidden_size, len(vocab))

        encoder_hidden_size = hidden_size*2 if bidirection else hidden_size
        decoder_hidden_size = hidden_size*2 if bidirection else hidden_size
        ### attention
        self.attn = attention.Attention(encoder_hidden_size, decoder_hidden_size)
        num = 4 if bidirection else 2
        self.Wa = nn.Linear(hidden_size*num, hidden_size)

    def forward(self, target, encoder_hidden, encoder_outputs):
        """
        :param target: shape:(batch, seq)
        :param encoder_hidden:
        :return:
        """
        # 1. encoder的hidden输出最为decoder的输入
        decoder_hidden = encoder_hidden  # (n_layer*direction, batch, hidden)

        batch_size = target.size(0)
        decoder_input = torch.LongTensor(torch.ones(
            [batch_size, 1], dtype=torch.int64)*self.vocab.SOS).to(config.device)  # (batch, 1)

        decoder_outputs = torch.zeros(
            [batch_size, self.max_seq_len+1, len(self.vocab)]).to(config.device)

        if random.random() > self.teach_forcing_rate: # teacher forcing 机制,加速收敛
            for t in range(self.max_seq_len+1):
                # decoder_output.shape: (batch, hidden_size)
                # decoder_hidden.shape: (n_layer*direction, batch, hidden_size)
                decoder_output_t, decoder_hidden = self.forward_step(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_outputs[:, t, :] = decoder_output_t
                decoder_input = target[:, t].unsqueeze(1)
        else:
            for t in range(self.max_seq_len+1):
                # decoder_output.shape: (batch, hidden_size)
                # decoder_hidden.shape: (n_layer*direction, batch, hidden_size)
                decoder_output_t, decoder_hidden = self.forward_step(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_outputs[:, t, :] = decoder_output_t
                # value.shape: (batch, 1)
                # index.shape: (batch, 1)
                value, index = torch.topk(decoder_output_t, k=1, dim=-1)
                decoder_input = index

        return decoder_outputs, decoder_hidden

    def forward_step(self, decoder_input, decoder_hidden, encoder_outputs):
        decoder_input_emb = self.embedding(decoder_input)  # (batch, 1, emb_size)

        # out.shape: (batch, 1, decode_hidden_size)
        # decoder_hidden: (num_layer, batch, hidden_size)
        out, decoder_hidden = self.gru(decoder_input_emb, decoder_hidden)

        # ATTENTION START
        attn_weight = self.attn(decoder_hidden, encoder_outputs).unsqueeze(1)  # (batch, 1, seq_len)
        context_vec = attn_weight.bmm(encoder_outputs)  # (batch, 1, encoder_hidden_size)
        concated = torch.cat([out, context_vec], dim=-1).squeeze(1)  # (batch, hidden_size(de+en))
        out = torch.tanh(self.Wa(concated))  # (batch, hidden_size)
        # ATTENTION END

        # out = out.squeeze(1)  # batch, hidden_size
        output = F.log_softmax(self.fc(out), dim=-1)  # batch, vocab_size
        return output, decoder_hidden

    def evaluate(self, encoder_outputs, encoder_hidden):
        decoder_hidden = encoder_hidden  # (n_layer*direction, batch, hidden)

        batch_size = encoder_hidden.size(1)
        decoder_input = torch.LongTensor(torch.ones(
            [batch_size, 1], dtype=torch.int64)*self.vocab.SOS).to(config.device)  # (batch, 1)

        indices = []
        for _ in range(self.max_seq_len+1):
            # decoder_output_t.shape: (batch, vocab_size)
            # decoder_hidden.shape: (n_layer*bidirection, batch, hidden_size)
            decoder_output_t, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs)
            # value.shape: (batch, 1)
            # index.shape: (batch, 1)
            value, index = torch.topk(decoder_output_t, k=1, dim=-1)
            decoder_input = index
            # if index.item() == vocab.EOS:
            #     break
            indices.append(index.squeeze(-1).cpu().detach().numpy())
        indices = np.array(indices).transpose()
        return indices  # (batch_size, max_seq_len)

    def evaluate_beam_search(self, encoder_outputs, encoder_hidden):
        """
        使用beam search进行evaluation
        :param encoder_outputs:
        :param encoder_hidden:
        :return:
        """
        batch_size = encoder_hidden.size(1)
        # 1. 构造第一次需要的输入数据，保存在堆中
        decoder_input = torch.LongTensor([[self.vocab.SOS]*batch_size]).to(config.device)
        decoder_hidden = encoder_hidden
        preve_beam = beam_search.BeamSearch()
        preve_beam.add(1, False, [decoder_input], decoder_input, decoder_hidden)
        while True:
            cur_beam = beam_search.BeamSearch()
            for _probility, _complete, _seq, _decoder_input, _decoder_hidden in preve_beam:
                if _complete == True:
                    cur_beam.add(_probility, _complete, _seq, _decoder_input, _decoder_hidden)
                else:
                    decoder_output_t, decoder_hidden = self.forward_step(_decoder_input, _decoder_hidden, encoder_outputs)
                    value, index = torch.topk(decoder_output_t, config.xhj_beam_width)
                    for m, n in zip(value[0], index[0]):
                        decoder_input = torch.LongTensor([[n]]).to(config.device)
                        seq = _seq + [n]
                        probility = _probility * m
                        if n.item() == self.vocab.EOS:
                            complete = True
                        else:
                            complete = False
                            # 把下一个时间步骤需要的输入保存到新的堆中
                            cur_beam.add(probility, complete, seq, decoder_input, decoder_hidden)

            best_prob, best_complete, best_seq, _, _ = max(cur_beam)
            if best_complete == True or len(best_seq)-1 == self.max_seq_len+1:
                return self._prepar_seq(best_seq)
            else:
                preve_beam = cur_beam

    def _prepar_seq(self, seq):
        """
        对结果进行基础处理，提供给后续转换为文字使用
        :param seq:
        :return:
        """
        if seq[0].item() == self.vocab.SOS:
            seq = seq[1:]
        if seq[-1].item() == self.vocab.EOS:
            seq = seq[:-1]
        seq = [i.item() for i in seq]
        return seq


