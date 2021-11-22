import torch
import torch.nn as nn
from model.base_lib.encoder import Encoder
from model.base_lib.decoder import Decoder
import config


class Seq2seq(nn.Module):
    """
    此处假设encoder和decoder的hidden_size num_layer  emb_size vocab都是相同的
    """
    def __init__(self, input_vocab, target_vocab, emb_size, num_layer, hidden_size, batch_first,
                 bidirection, teach_forcing_rate, decoder_max_seq_len):
        super(Seq2seq, self).__init__()
        self.encoder = Encoder(input_vocab, emb_size, num_layer, hidden_size, batch_first, bidirection)
        self.decoder = Decoder(target_vocab, emb_size, num_layer, hidden_size, batch_first, bidirection,
                               teach_forcing_rate, decoder_max_seq_len)

    def forward(self, input, target, input_len, target_len):
        encoder_outputs, encoder_hidden = self.encoder(input, input_len)
        decoder_outputs, decoder_hidden = self.decoder(target, encoder_hidden, encoder_outputs)
        return decoder_outputs, decoder_hidden

    def evaluate(self, inputs, input_length):
        with torch.no_grad():
            encoder_outputs, encoder_hidden = self.encoder(
                inputs, input_length)
            if config.beam_search:
                indices = self.decoder.evaluate_beam_search(encoder_outputs, encoder_hidden)
            else:
                indices = self.decoder.evaluate(encoder_outputs, encoder_hidden)
            return indices

