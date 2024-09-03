import os
from config import parsers
# transformer库是一个把各种预训练模型集成在一起的库，导入之后，你就可以选择性的使用自己想用的模型，这里使用的BERT模型。
# 所以导入了bert模型，和bert的分词器，这里是对bert的使用，而不是bert自身的源码。
from transformers import BertTokenizer,GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import re

class_dict={"体育":0,
            "娱乐":1,
            "财经":2,
            "房产":3,
            "家居":4,
            "教育":5,
            "科技":6,
            "时尚":7,
            "时政":8,
            "游戏":9,
            }
def read_data(file):
    # 读取文件
    all_data = open(file, "r", encoding="utf-8").read().split("\n")
    # 得到所有文本、所有标签、句子的最大长度
    texts, labels, max_length = [], [], []
    for data in all_data:
        if data:
            # text, label = data.split("\t")
            label, text = data.split("\t")
            label = class_dict[label]
            max_length.append(len(text))
            texts.append(text)
            labels.append(label)
    # 根据不同的数据集返回不同的内容
    if os.path.split(file)[1] == "cnews.train.txt":
        max_len = max(max_length)
        return texts, labels, max_len
    return texts, labels,

def read_data1(file):
    # 读取文件
    all_data = open(file, "r", encoding="utf-8").read().split("\n")
    # 得到所有文本、所有标签、句子的最大长度
    texts, labels, max_length = [], [], []
    for data in all_data:
        if data:
            label, text = get_text_label(data)
            max_length.append(len(text))
            texts.append(text)
            labels.append(label)
    # 根据不同的数据集返回不同的内容
    if os.path.split(file)[1] == "cnews.train.txt":
        max_len = max(max_length)
        return texts, labels, max_len
    return texts, labels,

def get_text_label(data):

    # 给定的文本
    input_text = """
    体育    黄蜂vs湖人首发：科比带伤战保罗 加索尔救赎之战 新浪体育讯北京时间4月27日，NBA季后赛首轮洛杉矶湖人主场迎战新奥尔良黄蜂，此前的比赛中，双方战成2-2平，因此本场比赛对于两支球队来说都非常重要，赛前双方也公布了首发阵容：湖人队：费舍尔、科比、阿泰斯特、加索尔、拜纳姆黄蜂队：保罗、贝里内利、阿里扎、兰德里、奥卡福[新浪NBA官方微博][新浪NBA湖人新闻动态微博][新浪NBA专题][黄蜂vs湖人图文直播室](新浪体育)
    """

    # 使用正则表达式提取标签和文本
    pattern = r"^(.*?)\s+(.*?)$"
    match = re.match(pattern, data.strip(), re.DOTALL)

    if match:
        label = match.group(1).strip()
        text = match.group(2).strip()
        print("Label:", label)
        print("Text:", text)
        return label, text
    else:
        print("No match found.")
        return None ,None


class MyDataset(Dataset):
    def __init__(self, texts, labels, max_length):
        self.all_text = texts
        self.all_label = labels
        self.max_len = max_length
        # self.tokenizer = GPT2Tokenizer.from_pretrained('gpt-small')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def __getitem__(self, index):
        # 取出一条数据并截断长度
        text = self.all_text[index][:self.max_len]
        label = self.all_label[index]

        # 分词
        text_id = self.tokenizer.tokenize(text)
        # 加上起始标志
        text_id = ["[CLS]"] + text_id

        # 编码
        token_id = self.tokenizer.convert_tokens_to_ids(text_id)
        # 掩码  -》
        mask = [1] * len(token_id) + [0] * (self.max_len + 2 - len(token_id))
        # 编码后  -》长度一致
        token_ids = token_id + [0] * (self.max_len + 2 - len(token_id))
        # str -》 int
        label = int(label)

        # 转化成tensor
        token_ids = torch.tensor(token_ids)
        mask = torch.tensor(mask)
        label = torch.tensor(label)

        return (token_ids, mask), label

    def __len__(self):
        # 得到文本的长度
        return len(self.all_text)


if __name__ == "__main__":
    train_text, train_label, max_len = read_data("./data/cnews.val.txt")
    print(train_text[0], train_label[0])
    trainDataset = MyDataset(train_text, train_label, max_len)
    trainDataloader = DataLoader(trainDataset, batch_size=3, shuffle=False)
    for batch_text, batch_label in trainDataloader:
        print(batch_text, batch_label)
