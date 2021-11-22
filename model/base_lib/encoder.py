import torch
import torch.nn as nn
from vocabs.vocab import Vocab


class Encoder(nn.Module):
    def __init__(self, vocab, emb_size, num_layer, hidden_size, batch_first=True, bidirection=False) -> None:
        super(Encoder, self).__init__()
        self.vocab = vocab
        self.emb_size = emb_size
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirection = bidirection

        self.embedding = nn.Embedding(
            num_embeddings=len(vocab), embedding_dim=emb_size, padding_idx=vocab.PAD)
        self.dropout = nn.Dropout(0.3)
        self.gru = torch.nn.GRU(input_size=emb_size, num_layers=num_layer,
                                hidden_size=hidden_size, batch_first=batch_first, bidirectional=bidirection)

    def forward(self, x, x_len):
        embeded = self.embedding(x)  # (batch, max_len, emb_size)
        embeded = self.dropout(embeded)
        embeded = torch.nn.utils.rnn.pack_padded_sequence(
            embeded, x_len, batch_first=self.batch_first)
        # output.shape: (batch, seq, hidden_size)
        # hidden.shape: (seq, batch, hidden_size)
        output, hidden = self.gru(embeded)
        # output.shape: (batch, seq, hidden_size)
        # output_len.shape: (batch,)
        output, output_len = torch.nn.utils.rnn.pad_packed_sequence(
            output, batch_first=self.batch_first, padding_value=self.vocab.PAD)

        return output, hidden

