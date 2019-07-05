#!/usr/bin/python
#coding:utf-8
import numpy as np
import pandas as pd
import jieba.posseg as pseg
import jieba
import re
import os
import codecs
from collections import defaultdict
from tqdm import tqdm
import pickle
import random
import argparse
import os


# load dictionary/names of all the corps  
# noted that jieba is Chinese text segmentation; see https://github.com/fxsjy/jieba
name = ["太平洋资产管理有限责任公司","张家港农商银行","江苏银行","中建投信托有限责任公司","华宝兴业基金"]
corps =set()
for i in range(len(name)):
    tmp_sheet = pd.read_excel("../dictionary/组合管理-持仓清单.xlsx",sheetname=name[i])
    corps = corps|set(list(tmp_sheet.loc[:,"主体名称"]))
    corps = corps|set(list(tmp_sheet.loc[:,"债券名称"]))
sheet2 = pd.read_excel('../dictionary/公司简称.xlsx',sheetname = 0)
corps = corps|(set(list(sheet2.iloc[:,1])))
corps = corps|(set(list(sheet2.iloc[:,0])))
jieba.load_userdict('../dictionary/mydict.txt')
jieba.load_userdict('../dictionary/negative.txt')
jieba.load_userdict('../dictionary/positive.txt')
jieba.load_userdict(corps)

# load negative words
neg_words = pd.read_excel("../dictionary/新闻负面词.xlsx")
jieba.load_userdict(list(neg_words.loc[:,"NewsClass"]))

# filter some nosiy data 
# pattern="[\s\.\!\/_,-:;~{}`^\\\[\]<=>?$%^*()+\"\']+|[+——！·【】‘’“”《》，。：；？、~@#￥%……&*（）]+0123456789qwertyuioplkjhgfdsazxcvbnm"
pattern="[\.\\/_,，.:;~{}`^\\\[\]<=>?$%^*()+\"\']+|[+·。：【】‘’“”《》、~@#￥%……&*（）]+0123456789"
pat = set(pattern)|set(["\n",'\u3000'," ","\s","","<br>"])
filterwords = ["<br>","责任编辑","DF","点击查看","热点栏目 资金流向 千股千评 个股诊断 最新评级 模拟交易 客户端","进入【新浪财经股吧】讨论","记者","鸣谢","报道","重点提示","重大事项","重要内容提示","提示：键盘也能翻页，试试“← →”键","原标题"]
# with codecs.open('../dictionary/stopwords_CN.dat','r') as fr:
#     stopwords=fr.readlines()
#     stopwords=[i.strip() for i in stopwords]
#     stopwords=set(stopwords)


# get test data
test_x, test_y = [],[]
#测试集每个分词后的句子对应的真实的句子，存在词典里面,
to_sentence = {}
#每个句子对应的文档index
to_document = {}
max_sent_in_doc = 30
max_word_in_sent = 45
UNKNOWN = 0
num_classes =2 
with open("../model/vocab.pickle",'rb') as f:
    vocab = pickle.load(f)
def FormData(sheet):
    for row in tqdm(range(len(sheet))):
        doc=np.zeros((30,45), dtype=np.int32)
        title = str(sheet.loc[row,"title"])
        text = str(sheet.loc[row,"content"])
        for item in filterwords:
            text = text.replace(item,"")
        sents = title +"。"+text
        count1 = 0
        for i, sent in enumerate(sents.split("。")):
            # filter the code in the news
            if "function()" in sent:
                continue
            if count1 < max_sent_in_doc:
                count = 0
                for j, word in enumerate(pseg.lcut(sent)):
                    kind = (list(word))[1][0]
                    tmpword = (list(word))[0]
                    if (tmpword not in pat) and (tmpword[0] not in pat) and (count < max_word_in_sent):
                        doc[count1][count] = vocab.get(tmpword, UNKNOWN)
                        count +=1
                to_sentence[str(doc[count1].tolist())] = sent
                count1 +=1
        # score==1: negative; score==0: positive
        # try:
        if sheet.loc[row,"score"]==0:
            label = 0
        else:
            label = 1
        labels = [0] * num_classes
        labels[label] = 1
        # except:
        #     labels = [0] * num_classes
        test_y.append(labels)
        test_x.append(doc.tolist())

# deal with every file in test_data 
path = "../test_data"
f = open("../output/deal_list","w")
for n_file in os.listdir(path):
    try:
        file_path = os.path.join(path,n_file)
        data = pd.read_excel(file_path)
        # dat = data.loc[data.clas=="财经网站"].copy()
        # print(len(dat))
        f.write(n_file+"\n")
        FormData(data)
    except:
        pass
f.close()
pickle.dump((to_sentence), open('../temp/to_sentence', 'wb'))
pickle.dump((test_x, test_y), open('../temp/test_data', 'wb'))
print("load test_data finished")
