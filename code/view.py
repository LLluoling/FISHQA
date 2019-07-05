#!/usr/bin/python
#coding:utf-8
import numpy as np
import pandas as pd
import pickle
#import matplotlib.pyplot as plt
#import seaborn as sns
from functools import reduce
from tqdm import tqdm
import os
import codecs
data_path = "../test_data"
output_path = "../output"
def union_f(x, y = ""):
    return x +" "+ y
with codecs.open("blacklist.txt",'r') as fr:
    b = fr.readlines()
blacklist = []
for i in b:
    blacklist.append(i.strip())
blacklist = set(blacklist)

f = open(os.path.join(output_path,"deal_list"))
title,content = [],[]
for line in f:
    filename = os.path.join(data_path,line.strip())
    sheet = pd.read_excel(filename)
    title.extend(list(sheet.loc[:,"title"]))
    content.extend(list(sheet.loc[:,"content"]))

# load att_words
with open('../temp/att_words.pickle', 'rb') as f:
    att_words = pickle.load(f)
# load att_sents
with open('../temp/att_sents.pickle', 'rb') as f:
    att_sents = pickle.load(f)
# load predict_y
with open('../temp/predict_y.pickle', 'rb') as f:
    y_pred = pickle.load(f)
# Y = [i.index(1) for i in y_pred]
with open('../temp/to_sentence', 'rb') as f:
    to_sentence = pickle.load(f)
with open('../temp/test_data', 'rb') as f:
    X,_ = pickle.load(f)

with open('../model/vocab.pickle', 'rb') as f:
    vocab = pickle.load(f)

new_vocab = dict(map(lambda t:(t[1],t[0]), vocab.items()))

print("title,content,att_words,att_sents,y_pred: ",len(title),len(content),len(att_words),len(att_sents),len(y_pred))

# output a file to view attended sentences
S,W = [],[]
for doc in tqdm(range(len(att_sents))):
    b = []
    for query in range(len(att_sents[0])):
        b.append(sorted(range(len(att_sents[doc][query])), key=list(att_sents[doc][query]).__getitem__,reverse=True))
    # load sents
    try:
        tmp = ""
        count = 0;i = 0
        for i in range(3):
            temp_list = []
            for sent in b[i]:
                if len(to_sentence[str(X[doc][sent])])>3:
                    temp_list.append(to_sentence[str(X[doc][sent])])
                    count += 1
                if count >=3:
                    break
            tmp += reduce(union_f,temp_list)+"||"
        S.append(tmp)
    except:
        S.append(title[doc])
    # load words
    try:
        tmp = ""
        for i in range(30):
            word = new_vocab[X[doc][i][att_words[doc][i]]]
            if word!="UNKNOW_TOKEN":
                tmp += word+"  "
        W.append(tmp)
    except:
        pass


writer = pd.ExcelWriter('../output/result.xlsx')
df = pd.DataFrame(columns=['title','content',"predict_score","attened_sents","attened_words"])
df.loc[:,"title"] = title
df.loc[:,"content"] = content
df.loc[:,"predict_score"] = list(y_pred)
df.loc[:,"attened_sents"] = S
df.loc[:,"attened_words"] = W

df.to_excel(writer,'Sheet1')
writer.save()
print("done!")
