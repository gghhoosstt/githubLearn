import torch as t
import math
from torch.nn import functional as f
# a = t.randn(3,2)
# b = f.softmax(a,dim = 1)
# featurematrix = torch.zeros((4,768))
# featurematrix = featurematrix.float()  # 转化成浮点型tensor
#
# file1 = open("C:\\Users\\no_password\\Desktop\\fsdownload\\error.txt","r",encoding = 'utf-8')
# file = open("F:\\PYTHONproject\\EAAI-25-master\\testdata\\sciEntsBank\\2way\\test-unseen-answers.txt","r",encoding = 'utf-8')
# file2 = open("C:\\Users\\no_password\\Desktop\\fsdownload\\corr.txt","w")
# lines = file.readlines()
# error = file1.readlines()
# i = 0
# ids = []
# for line in error:
#     label = line.strip().split("条")[0]
#     label = label.strip().split("第")[1]
#     ids.append(int(label))
# print(ids)
# for line in lines:
#     if len(ids)> 0:
#         if i != ids[0]:
#             line = line.strip().split('\t')
#             line = line[2]+"***"+line[4]+"||label:"+line[5]
#             file2.write("第%d条:%s\n"%(i,line))
#         else:
#              ids.pop(0)
#     i += 1

# print("label =1:",cor)
# print("label =0:",incor)
# file.close()
# file1.close()
# file2.close()
