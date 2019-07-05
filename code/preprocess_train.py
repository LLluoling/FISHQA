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
from model import shuffle_data


# load dictionary/names of all the corps  
# noted that jieba is Chinese text segmentation; see https://github.com/fxsjy/jieba
name = ["太平洋资产管理有限责任公司","张家港农商银行","江苏银行","中建投信托有限责任公司","华宝兴业基金"]
corps =set()
for i in range(len(name)):
    sheet = pd.read_excel("../dictionary/组合管理-持仓清单.xlsx",sheetname=name[i])
    corps = corps|set(list(sheet.loc[:,"主体名称"]))
    corps = corps|set(list(sheet.loc[:,"债券名称"]))
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

# filter some nosiy marks 
pattern="[\.\\/_,，.:;~{}`^\\\[\]<=>?$%^*()+\"\']+|[+·。：【】‘’“”《》、~@#￥%……&*（）]+0123456789"
pat = set(pattern)|set(["\n",'\u3000'," ","\s","","<br>"])

# some noise words in chinese news
filterwords = ["<br>","责任编辑","DF","点击查看","热点栏目 资金流向 千股千评 个股诊断 最新评级 模拟交易 客户端","进入【新浪财经股吧】讨论","记者","鸣谢","报道","重点提示","重大事项","重要内容提示","提示：键盘也能翻页，试试“← →”键","原标题"]
# with codecs.open('../dictionary/stopwords_CN.dat','r') as fr:
#     stopwords=fr.readlines()
#     stopwords=[i.strip() for i in stopwords]
#     stopwords=set(stopwords)


# count the frequency of each word in documents
print("count word frequency")
word_freq = defaultdict(int)
def Getdata(sheet):
    for row in tqdm(range(len(sheet))):
        title=str(sheet.loc[row,"title"])
        content=str(sheet.loc[row,"content"])
        for item in filterwords:
            content = content.replace(item,"")
        sents = title +"　" +content
        words=pseg.lcut(sents)
        for j, word in enumerate(words):
            kind = (list(word))[1][0]
            tmpword = (list(word))[0]
            #if (kind not in ['e','x','m','u']) and (tmpword not in stopwords):
            if (tmpword not in pat) and (tmpword[0] not in pat):
                word_freq[tmpword]+=1
path = "../training_data"
for n_file in os.listdir(path):
    file_path = os.path.join(path,n_file)
    sheet = pd.read_excel(file_path)
    Getdata(sheet)

# count the frequency of each word in query set
q=[]
f = open("../Query")
for line in f:
    q.append(line)
    words=pseg.lcut(str(line.strip()))
    for j, word in enumerate(words):
        kind = (list(word))[1][0]
        tmpword = (list(word))[0]
        if (tmpword not in pat) and (tmpword[0] not in pat):
            word_freq[tmpword]+=1
f.close()
print("previous data length:",len(word_freq))


# load word frequency
if not os.path.exists("../model"):
    os.mkdir("../model")
with open('../model/word_freq.pickle', 'wb') as g:
    pickle.dump(word_freq, g)
    print(len(word_freq),"word_freq save finished")            
# sorted by word frequency and remove those whose frquency < 3 
sort_words = list(sorted(word_freq.items(), key=lambda x:-x[1]))
print("the 10 most words:",sort_words[:10],"\n the 10 least words:",sort_words[-10:])


# load word vocab
vocab = {}
i = 3
vocab['UNKNOW_TOKEN'] = 0

for word, freq in sort_words:
    if freq > 3:
        vocab[word] = i
        i += 1
with open('../model/vocab.pickle', 'wb') as f:
    pickle.dump(vocab, f)
    print(len(vocab),"vocab save finished")   
UNKNOWN = 0
num_classes = 2


# get training data
data_x,data_y =[],[]
max_sent_in_doc = 30
max_word_in_sent = 45

# we form 3 queries for our model (depending on your datasets and your need)
question = np.zeros((3,max_word_in_sent), dtype=np.int32)

for i,ite in enumerate(q):
    words=pseg.lcut(ite)
    count = 0
    for j, word in enumerate(words):
        kind = (list(word))[1][0]
        tmpword = (list(word))[0]
        if (tmpword not in pat) and (tmpword[0] not in pat):
            question[i][count] = vocab.get(tmpword, UNKNOWN)
            count +=1
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
                count1 +=1
        # 0: non-neg 1: neg
        if sheet.loc[row,"score"]==0:
            label = 0
        else:
            label = 1
        labels = [0] * num_classes
        labels[label] = 1
        data_y.append(labels)
        data_x.append(doc.tolist())
for n_file in os.listdir(path):
    file_path = os.path.join(path,n_file)
    sheet = pd.read_excel(file_path)
    FormData(sheet)
print("load train_data finished, length: ",len(data_x))


# load training data
data_x,data_y = shuffle_data(data_x,data_y)
train_x,train_y,eval_x,eval_y = [],[],[],[]
for i in range(len(data_x)):
    r = random.random()
    if r<0.8:
        train_x.append(data_x[i])
        train_y.append(data_y[i])
    else:
        eval_x.append(data_x[i])
        eval_y.append(data_y[i])

print("shuffle data finished!")
pickle.dump((train_x,train_y), open('../model/train_data', 'wb'))
pickle.dump((eval_x,eval_y), open('../model/dev_data', 'wb'))
pickle.dump((question[0].tolist()), open('../model/q1_data', 'wb'))
pickle.dump((question[1].tolist()), open('../model/q2_data', 'wb'))
pickle.dump((question[2].tolist()), open('../model/q3_data', 'wb'))
print("store training data finished!")
