#!/usr/bin/python
#coding:utf-8
import tensorflow as tf
import time
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
import random
import pandas as pd

# return the length of each sequence
def length(sequences):
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(seq_len, tf.int32)
# load 3 query set (we set 3 based on )
def read_question():
    with open('../model/q1_data', 'rb') as f:
        q1 = pickle.load(f)
    with open('../model/q2_data', 'rb') as g:
        q2 = pickle.load(g)
    with open('../model/q3_data', 'rb') as b:
        q3 = pickle.load(b)
        return [q1,q2,q3]
def shuffle_data(x,y):
    train_x = [];train_y=[]
    li = np.random.permutation(len(x))
    for i in tqdm(range(len(li))):
        train_x.append(x[li[i]])
        train_y.append(y[li[i]])
    return train_x,train_y
class FISHQA():

    def __init__(self, vocab_size, num_classes, embedding_size=200, hidden_size=50, dropout_keep_proba=0.5,query=[]):

        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dropout_keep_proba = dropout_keep_proba
        self.query = query

        with tf.name_scope('placeholder'):
            self.max_sentence_num = tf.placeholder(tf.int32, name='max_sentence_num')
            self.max_sentence_length = tf.placeholder(tf.int32, name='max_sentence_length')
            self.batch_size = tf.placeholder(tf.int32, name='batch_size')
            #x shape [batch_size, sentence_num,word_num ]
            #y shape [batch_size, num_classes]
            self.input_x = tf.placeholder(tf.int32, [None, None, None], name='input_x')
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
            self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        word_embedded,q1_emb,q2_emb,q3_emb = self.word2vec()
        sent_vec,att_word = self.sent2vec(word_embedded,q1_emb,q2_emb,q3_emb)
        doc_vec,att_sent = self.doc2vec(sent_vec,q1_emb,q2_emb,q3_emb)
        out = self.classifer(doc_vec)

        self.out = out
        self.att_word = att_word
        self.att_sent = att_sent
    def word2vec(self):
        with tf.name_scope("embedding"):
            embedding_mat = tf.Variable(tf.truncated_normal((self.vocab_size, self.embedding_size)))
            #shape: [batch_size, sent_in_doc, word_in_sent, embedding_size]
            # 45 is the max
            word_embedded = tf.nn.embedding_lookup(embedding_mat, self.input_x)
            q1_emb = tf.reduce_sum(tf.nn.embedding_lookup(embedding_mat, self.query[0]),axis=0)/45
            q2_emb = tf.reduce_sum(tf.nn.embedding_lookup(embedding_mat, self.query[1]),axis=0)/45
            q3_emb = tf.reduce_sum(tf.nn.embedding_lookup(embedding_mat, self.query[2]),axis=0)/45
        return word_embedded,q1_emb,q2_emb,q3_emb

    def sent2vec(self, word_embedded,q1_emb,q2_emb,q3_emb):
        with tf.name_scope("sent2vec"):
            #GRU input size : [batch_size, max_time, ...]
            #shape: [batch_size*sent_in_doc, word_in_sent, embedding_size]
            word_embedded = tf.reshape(word_embedded, [-1, self.max_sentence_length, self.embedding_size])
            #shape: [batch_size*sent_in_doce, word_in_sent, hidden_size*2]
            word_encoded = self.BidirectionalGRUEncoder(word_embedded, name='word_encoder')
            #shape: [batch_size*sent_in_doc, hidden_size*2]
            sent_temp,att_word = self.AttentionLayer(word_encoded,q1_emb,q2_emb,q3_emb, name='word_attention')
            sent_vec = layers.dropout(sent_temp, keep_prob=self.dropout_keep_proba,is_training=self.is_training,)
            return sent_vec,att_word

    def doc2vec(self, sent_vec,q1_embedded,q2_embedded,q3_embedded):
        # the same with sent2vec
        with tf.name_scope("doc2vec"):
            sent_vec = tf.reshape(sent_vec, [-1, self.max_sentence_num, self.hidden_size*2])
            #shape为[batch_size, sent_in_doc, hidden_size*2]
            doc_encoded = self.BidirectionalGRUEncoder(sent_vec, name='sent_encoder')
            #shape为[batch_szie, hidden_szie*2]
            doc_temp,att_sent = self.SentenceAttentionLayer(doc_encoded,q1_embedded,q2_embedded,q3_embedded,name='sent_attention')
            doc_vec = layers.dropout(doc_temp, keep_prob=self.dropout_keep_proba,is_training=self.is_training,)
            return doc_vec,att_sent

    def classifer(self, doc_vec):
        with tf.name_scope('doc_classification'):
            out = layers.fully_connected(inputs=doc_vec, num_outputs=self.num_classes, activation_fn=None)
            return out

    def BidirectionalGRUEncoder(self, inputs, name):
        #inputs shape: [batch_size, max_time, voc_size]
        with tf.variable_scope(name):
            GRU_cell_fw = rnn.GRUCell(self.hidden_size)
            GRU_cell_bw = rnn.GRUCell(self.hidden_size)
            #fw_outputs, bw_outputs size: [batch_size, max_time, hidden_size]
            # time_major=False,
            # if time_major = True, tensor shape: `[max_time, batch_size, depth]`.
            # if time_major = False, tensor shape`[batch_size, max_time, depth]`.
            ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                                 cell_bw=GRU_cell_bw,
                                                                                 inputs=inputs,
                                                                                 sequence_length=length(inputs),
                                                                                 dtype=tf.float32)
            #outputs size [batch_size, max_time, hidden_size*2]
            outputs = tf.concat((fw_outputs, bw_outputs), 2)
            return outputs

    def AttentionLayer(self, inputs, q1_emb,q2_emb,q3_emb,name):
        #inputs size [batch_size, max_time, encoder_size(hidden_size * 2)]
        with tf.variable_scope(name):
            # u_context length is 2×hidden_size
            u_context = tf.Variable(tf.truncated_normal([self.hidden_size * 2]), name='u_context')
            # output size [batch_size, max_time, hidden_size * 2]
            h1 = layers.fully_connected(inputs, self.hidden_size * 2, activation_fn=tf.nn.tanh)
            h2 = layers.fully_connected(inputs, self.hidden_size * 2, activation_fn=tf.nn.tanh)
            h3 = layers.fully_connected(inputs, self.hidden_size * 2, activation_fn=tf.nn.tanh)
            h4 = layers.fully_connected(inputs, self.hidden_size * 2, activation_fn=tf.nn.tanh)

            # shape [batch_size, max_time, 1]
            t_alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h1, u_context), axis=2, keep_dims=True), dim=1)
            q_alpha1 = tf.nn.softmax(tf.reduce_sum(tf.multiply(h2, q1_emb), axis=2, keep_dims=True), dim=1)
            q_alpha2 = tf.nn.softmax(tf.reduce_sum(tf.multiply(h3, q2_emb), axis=2, keep_dims=True), dim=1)
            q_alpha3 = tf.nn.softmax(tf.reduce_sum(tf.multiply(h4, q3_emb), axis=2, keep_dims=True), dim=1)

            alpha = (t_alpha+q_alpha1+q_alpha2+q_alpha3)/4

            a = tf.nn.top_k((tf.reshape(alpha,[-1,self.max_sentence_length])),k=1).indices
            # shape [batch_size, max_time, 1]
            # alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
            # reduce_sum  [batch_size, max_time, hidden_size*2] ---> [batch_size, hidden_size*2]
            atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
            # atten_output = tf.reduce_sum(inputs,axis=1)
            return atten_output,a
    def SentenceAttentionLayer(self, inputs,q1_emb,q2_emb,q3_emb, name):
        # inputs size [batch_size, max_time, encoder_size(hidden_size * 2)]
        with tf.variable_scope(name):
            u_context = tf.Variable(tf.truncated_normal([self.hidden_size * 2]), name='u_context')

            h1 = layers.fully_connected(inputs, self.hidden_size * 2, activation_fn=tf.nn.tanh)
            h2 = layers.fully_connected(inputs, self.hidden_size * 2, activation_fn=tf.nn.tanh)
            h3 = layers.fully_connected(inputs, self.hidden_size * 2, activation_fn=tf.nn.tanh)
            h4 = layers.fully_connected(inputs, self.hidden_size * 2, activation_fn=tf.nn.tanh)

            # shape [batch_size, max_time, 1]
            t_alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h1, u_context), axis=2, keep_dims=True), dim=1)
            q_alpha1 = tf.nn.softmax(tf.reduce_sum(tf.multiply(h2, q1_emb), axis=2, keep_dims=True), dim=1)
            q_alpha2 = tf.nn.softmax(tf.reduce_sum(tf.multiply(h3, q2_emb), axis=2, keep_dims=True), dim=1)
            q_alpha3 = tf.nn.softmax(tf.reduce_sum(tf.multiply(h4, q3_emb), axis=2, keep_dims=True), dim=1)

            # sents shape [batch_size, sent_in_doc, hidden_size*2]

            alpha = (t_alpha+q_alpha1+q_alpha2+q_alpha3)/4
            #tf.add_to_collection('attention_value',alpha)
            #reduce_sum [batch_szie, max_time, hidden_szie*2] ---> [batch_size, hidden_size*2]
            atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)

            att = tf.concat([t_alpha,q_alpha1,q_alpha2,q_alpha3],0)
            #atten_output = tf.reduce_sum(inputs,axis=1)
            return atten_output,att
