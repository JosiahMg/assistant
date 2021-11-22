"""

"""
import torch
import config
from vocabs.vocab import Vocab
from torch.utils.data import DataLoader, Dataset


xhj_vocab = Vocab().load_vocab(config.xhj_vocab_path)


class XhjChatbotDataset(Dataset):
    def __init__(self):
        self.input_path = config.mid_xiaohuangji_question_path
        self.target_path = config.mid_xiaohuangji_answer_path
        self.input_lines = open(self.input_path, encoding='utf-8').readlines()
        self.target_lines = open(
            self.target_path, encoding='utf-8').readlines()
        assert len(self.input_lines) == len(
            self.target_lines), "input and target length not equal"
        self.vocab = xhj_vocab

    def __getitem__(self, item):
        inputs = self.input_lines[item].strip().split()
        targets = self.target_lines[item].strip().split()
        input_len = len(inputs)
        target_len = len(targets)
        return inputs, targets, input_len, target_len

    def __len__(self):
        return len(self.input_lines)

    def collate_fn(self, batch):
        """

        :param batch: [(input, target, input_len, target_len), (), ...]
        :return:
        """
        batch = sorted(batch, key=lambda x: x[-2], reverse=True)
        inputs, target, input_len, target_len = zip(*batch)

        inputs = [self.vocab.transform(
            i, max_len=config.xhj_max_encoder_len) for i in inputs]
        inputs = torch.LongTensor(inputs)

        target = [self.vocab.transform(
            i, max_len=config.xhj_max_decoder_len, add_eos=True) for i in target]
        target = torch.LongTensor(target)

        input_len = [input_l if input_l < config.xhj_max_encoder_len else config.xhj_max_encoder_len for input_l in input_len]
        target_len = [target_l if target_l < config.xhj_max_decoder_len else config.xhj_max_decoder_len for target_l in target_len]

        input_len = torch.LongTensor(input_len)
        target_len = torch.LongTensor(target_len)

        return inputs, target, input_len, target_len


xhj_dataset = XhjChatbotDataset()
train_data_loader = DataLoader(
    xhj_dataset, batch_size=config.xhj_batch_size, shuffle=True, collate_fn=xhj_dataset.collate_fn)

if __name__ == '__main__':
    for idx, (input, target, input_len, target_len) in enumerate(train_data_loader):
        print(idx)
        print(input.shape)
        print(target.shape)
        print(input_len)
        print(target_len)
        for data in input.tolist():
            print(' '.join(config.xhj_vocab.inverse_transform(data)))
        print('-----------------------------')
        for data in target.tolist():
            print(' '.join(config.xhj_vocab.inverse_transform(data)))
        break
