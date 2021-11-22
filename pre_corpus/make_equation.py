import sys
from pprint import pprint
import numpy as np
import pandas as pd
import config
import re
import os


def move_redundant(equation):
    return re.sub(r'\+*-+\+*-*', r'-', equation)


def create_equation():
    res = ''
    for i in range(7):
        num = np.random.randint(0, 1000)
        res += str(num)
        oper = np.random.choice(['+', '-', '+', '-', 's'])
        if oper == 's':
            break
        res += oper
    if res[-1] in '+-':
        res += str(np.random.randint(0, 100))
    return res


def make_equation(filename):
    if os.path.exists(filename):
        df = pd.read_csv(filename, encoding='utf-8')
    else:
        df = pd.DataFrame(columns=['question', 'answer', 'equation'])
    equation = create_equation()

    # print(equation)
    df_dict = {}

    try:
        res = eval(equation)
        df_dict['question'] = equation
        df_dict['answer'] = res
        df_dict['equation'] = equation
        df = df.append([df_dict])
        # print(df_dict)
        df.to_csv(filename, encoding='utf-8', index=False)
    except:
        pass


from tqdm import tqdm

if __name__ == '__main__':
    for i in tqdm(range(5000)):
        if i < 5000:
            make_equation(config.mid_math_oper_test_path)
        make_equation(config.mid_math_oper_train_path)


