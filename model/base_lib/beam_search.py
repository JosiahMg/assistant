"""
Beam Search

1. Beam Search 是介于贪心算法和计算全部概率之间的一种束搜索的方法，假设BeamSearch=2,表示每次
保存的最大的概率数为2个，这里每次保存两个，下一个时间步骤一样，也保存两个，这样就可以达到约束搜
索空间大小的目的，从而提高算法的效率；
2. BeamSearch=1时，就是贪心算法，beam width=候选词的时候，就是计算全部概率，beam width为超参数；
3. 堆： 优先级队列，先进先出
4. 使用堆完成Beam Search
    a. 构造<SOS>开始符号等第一次输入的信息，保存在堆中;
    b. 取出堆中的数据，进行forward_step操作，获得当前时间步的output和hidden
    c. 从output中选择topk(k=beam width)个输出，作为下一次的input
    d. 把下一个时间步需要的输入等数据保存在一个新的堆中
    e. 获取新的堆中的优先级最高(概率最大)的数据，判断数据是否是EOS结尾，是否达到最大长度，如果是则停止迭代
    f. 如果不是，则重新遍历新的堆中的数据

"""
import config
import heapq


class BeamSearch:
    def __init__(self):
        self.heap = list()
        self.beam_width = config.xhj_beam_width

    def add(self, probility, complete, seq, decoder_input, decoder_hidden):
        """
        添加数据，同时判断总的数据个数，多则删除
        :param probility: 概率乘积
        :param complete: 最后一个是否为EOS
        :param seq: List 所有token的列表
        :param decoder_input: 下一次进行编码的输入，通过前一次获取
        :param decoder_hidden: 下一次进行编码的hidden,通过前一次获得
        :return:
        """
        heapq.heappush(self.heap, [probility, complete, seq, decoder_input, decoder_hidden])
        # 判断数据的个数，如果大则弹出，保证总个数小于等于beam width
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)

    def __iter__(self):
        return iter(self.heap)


