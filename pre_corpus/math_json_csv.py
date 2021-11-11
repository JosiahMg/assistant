"""
功能说明:
将math json格式转换成csv格式:


输入数据格式:
{   "id": "1099539",
    "segmented_text": "五 年 级 同 学 参 加 义 务 捐 书 活 动 ， 五 1 班 捐 了 500 本 ， 五 2 班 捐 的 本 数 是 五 1 班 80% ， 五 3 班 捐 的 本 数 是 五 2 班 120% ， 五 1 班 和 五 3 班 比 谁 捐 书 多 ？ ( 请 用 两 种 方 法 比 较 一 下 ) ．",
    "original_text": "五年级同学参加义务捐书活动，五1班捐了500本，五2班捐的本数是五1班80%，五3班捐的本数是五2班120%，五1班和五3班比谁捐书多？(请用两种方法比较一下)．",
    "ans": "1",
    "equation": "x=1"
}

输出数据格式:
[id, question, answer, equation]

用途:
用于数学应用题解答的数据集

"""

import config
import json
import pandas as pd
import re
from pprint import pprint


def is_equal(a, b):
    """比较两个结果是否相等
    """
    a = round(float(a), 6)
    b = round(float(b), 6)
    return a == b


def remove_bucket(equation):
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
            if is_equal(eval(new_equation.replace(' ', '')), eval_equation):
                equation = new_equation
        except:
            pass
    return equation.replace(' ', '')


def load_data(filename):
    """读取训练数据，并做一些标准化，保证equation是可以eval的
    参考：https://kexue.fm/archives/7809
    """
    df = pd.DataFrame(columns=['question', 'answer', 'equation'])
    for l in open(filename, encoding='utf-8'):
        l = json.loads(l)
        question, equation, answer = l['original_text'], l['equation'], l['ans']
        # 处理带分数
        question = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', question)
        equation = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', equation)
        answer = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', answer)
        equation = re.sub('(\d+)\(', '\\1+(', equation)
        answer = re.sub('(\d+)\(', '\\1+(', answer)
        # 分数去括号
        question = re.sub('\((\d+/\d+)\)', '\\1', question)
        # 处理百分数
        equation = re.sub('([\.\d]+)%', '(\\1/100)', equation)
        answer = re.sub('([\.\d]+)%', '(\\1/100)', answer)
        # 冒号转除号、剩余百分号处理
        equation = equation.replace(':', '/').replace('%', '/100')
        answer = answer.replace(':', '/').replace('%', '/100')
        if equation[:2] == 'x=':
            equation = equation[2:]
        try:
            if is_equal(eval(equation), eval(answer)):
                df_elem = {'question': question, 'answer': answer,
                           'equation': remove_bucket(equation)}
                df = df.append([df_elem])
        except:
            continue
    df.to_csv(config.math_ape_path, encoding='utf-8', index=False, sep=',')
    return df


pprint(load_data(config.json_math_test_path))