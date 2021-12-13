import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import config
from vocabs.vocab import Vocab


que_vocab = Vocab().load_vocab(config.math_vocab_question_path)
equ_vocab = Vocab().load_vocab(config.math_vocab_equation_path)


class MathDataset(nn.Module):
    def __init__(self, filename):
        super(MathDataset, self).__init__()
        self.df_math = pd.read_csv(filename, encoding='utf-8', sep=',')

    def __getitem__(self, item):
        que = list(self.df_math.iloc[item]['question'])
        equ = list(self.df_math.iloc[item]['equation'])
        try:
            ans = float(self.df_math.iloc[item]['answer'])
        except:
            print(self.df_math.iloc[item]['answer'])

        return que, equ, ans, len(que), len(equ)

    def __len__(self):
        return len(self.df_math)


def collate_fn(batch):
    # 排序
    batch = sorted(batch, key=lambda x: x[-2], reverse=True)

    ques, equs, ans, ques_len, ans_len = zip(*batch)

    ques = [que_vocab.transform(i, max_len=config.math_max_ques_len) for i in ques]
    ques = torch.LongTensor(ques)

    equs = [equ_vocab.transform(i, max_len=config.math_max_equa_len, add_eos=True) for i in equs]
    equs = torch.LongTensor(equs)

    ques_len = [elem if elem < config.math_max_ques_len else config.math_max_ques_len for elem in ques_len]
    ans_len = [elem if elem < config.xhj_max_decoder_len else config.math_max_equa_len for elem in ans_len]

    ques_len = torch.LongTensor(ques_len)
    ans_len = torch.LongTensor(ans_len)

    return ques, equs, ans, ques_len, ans_len


trian_dataset = MathDataset(config.mid_math_ape_valid_path)
trian_dataloader = DataLoader(trian_dataset, batch_size=config.math_batch_size, shuffle=True, collate_fn=collate_fn)

valid_dataset = MathDataset(config.mid_math_ape_valid_path)
valid_dataloader = DataLoader(valid_dataset, batch_size=config.math_batch_size, shuffle=False, collate_fn=collate_fn)

test_dataset = MathDataset(config.mid_math_ape_test_path)
test_dataloader = DataLoader(test_dataset, batch_size=config.math_batch_size, shuffle=False, collate_fn=collate_fn)

if __name__ == '__main__':
    for i, (inputs, targets, ans, input_lens, target_lens) in enumerate(trian_dataloader):
        print(inputs.shape)
        print(targets.shape)
        print(ans)
        print(input_lens)
        print(target_lens)
        break

