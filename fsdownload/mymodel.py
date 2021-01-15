import torch as t
import logging
import os

import numpy as np
import torch

from torch import nn
from torch.nn import functional as f

from pytorch_pretrained_bert import file_utils#又见了，小伙子
from pytorch_pretrained_bert import modeling
from pytorch_pretrained_bert import tokenization


def align(lastEmbedding, index):
    r"""
    ```python
    lastEmbedding: torch.FloatTensor, batchsize * sequencelen * hiddensize
    index: batchsize * 2
    第i条RA索引左闭右开： 1 ~ sep[i][0]
    第i条A索引左闭右开：sep[i][0]+1 ~ sep[i][1]
    RA: LEN(RA)* hiddensize
    A: LEN(A)* hiddensize
    retrun：des : batchsize *  hiddensize
    """
    des = torch.zeros((len(lastEmbedding), 768))
    des = des.float()  # 转化成浮点型tensor
    des = des.cuda()
    for i in range(len(index)):
        # raindex = 17
        # aindex = 54
        raindex = index[i][0]
        aindex = index[i][1]
        RA = lastEmbedding[i][1:raindex]
        A = lastEmbedding[i][raindex + 1:aindex]
        alignMatrix = torch.mm(A, RA.T)
        # average pooling
        raweight = torch.mean(alignMatrix, 0)  # 列均值，得到列维度的一维数组
        raweight = f.softmax(raweight, dim=0)
        raweight = raweight.unsqueeze(0)
        aweight = torch.mean(alignMatrix, 1)  # 行均值，得到行维度的一维数组。
        aweight = f.softmax(aweight, dim=0)
        aweight = aweight.unsqueeze(0)  # 在0维扩展，变成1 * len(A)
        avector = torch.mm(aweight, A)
        ravector = torch.mm(raweight, RA)
        # if(w == "test"):
        #     filew = os.path.join("output", "weight.txt")
        #     with open(filew, "a+") as writer:
        #         writer.write("第%d个\n" % (i+1))
        #         writer.write("aweight:%s\n"%str(aweight))
        #         writer.write("raweight:%s\n"%str(raweight))
        des[i] = avector + ravector
    return des

class MyAlign(modeling.BertPreTrainedModel):

    sepindex = []
    batch = 0
    step = 0

    def __init__(self,bert_location,nway):
        super(MyAlign, self).__init__(bert_location)
        self.bert = modeling.BertModel(bert_location)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768,nway)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        #这是对齐版本
        last_embedding, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        start = self.step * self.batch
        mycls = align(last_embedding, self.sepindex[start:start + len(last_embedding)])
        mycls = self.dropout(mycls)
        logits = self.classifier(mycls)
        #下面是bertforsequenceclassification 的版本，相当于1.0Bert版本。
        # _, pooled_out = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # pooled_out = self.dropout(pooled_out)
        # logits = self.classifier(pooled_out)

        return logits
