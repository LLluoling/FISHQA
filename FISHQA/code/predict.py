#!/usr/bin/python
#coding:utf-8
from model import FISHQA,read_question,shuffle_data
import tensorflow as tf
import time
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
import random
import pandas as pd
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--logdir', default='1526700733')
args = parser.parse_args()
# Data loading params
tf.flags.DEFINE_string("data_dir", "../data/data.dat", "data directory")
tf.flags.DEFINE_integer("vocab_size", 52812, "vocabulary size")
tf.flags.DEFINE_integer("num_classes", 2, "number of classes")
tf.flags.DEFINE_integer("embedding_size", 200, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_integer("hidden_size", 100, "Dimensionality of GRU hidden layer (default: 50)")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("evaluate_every", 10, "evaluate every this many batches")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.flags.DEFINE_float("grad_clip", 5, "grad clip to prevent gradient explode")
tf.flags.DEFINE_float("sentence_num", 30, "the max number of sentence in a document")
tf.flags.DEFINE_float("sentence_length", 45, "the max length of each sentence")

with open("../temp/test_data", 'rb') as f:
    test_x,test_y = pickle.load(f)

FLAGS = tf.flags.FLAGS
print("loading test data finished")

def main():
    with tf.Session() as sess: 
        fishqa = FISHQA(vocab_size=FLAGS.vocab_size,
                        num_classes=FLAGS.num_classes,
                        embedding_size=FLAGS.embedding_size,
                        hidden_size=FLAGS.hidden_size,
                        dropout_keep_proba=0.5,
                        query = read_question()
                        )
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=fishqa.input_y,
                                                                          logits=fishqa.out,
                                                                          name='loss'))
        with tf.name_scope('accuracy'):
            predict = tf.argmax(fishqa.out, axis=1, name='predict')
            label = tf.argmax(fishqa.input_y, axis=1, name='label')
            acc = tf.reduce_mean(tf.cast(tf.equal(predict, label), tf.float32))
        
        with tf.name_scope('att_words'):
            att_words = tf.reshape(fishqa.att_word,[-1,30])
        with tf.name_scope('att_sents'):
            att_sents = tf.reshape(fishqa.att_sent,[-1,4,30])
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", args.logdir))


        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_path = checkpoint_dir + '/my-model.ckpt'

        # saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
        def test_step(x, y):
            predictions,labels  = [],[]
            attend_w,attend_s = [],[]
            for i in range(0, len(x), FLAGS.batch_size):

                feed_dict = {
                    fishqa.input_x: x[i:i + FLAGS.batch_size],
                    fishqa.input_y: y[i:i + FLAGS.batch_size],
                    fishqa.max_sentence_num: 30,
                    fishqa.max_sentence_length: 45,
                    fishqa.batch_size: 64,
                    fishqa.is_training:False
                }
                # step, summaries,cost, accuracy,correctNumber = sess.run([global_step, dev_summary_op,loss,acc,accNUM], feed_dict)
                pre,att_w,att_s= sess.run([predict,att_words,att_sents], feed_dict)
                attend_w.extend(att_w)
                attend_s.extend(att_s)
                predictions.extend(pre)

            print("predict score done!")
            pickle.dump(attend_w, open('../temp/att_words.pickle', 'wb'))
            pickle.dump(attend_s, open('../temp/att_sents.pickle', 'wb'))
            pickle.dump(predictions, open('../temp/predict_y.pickle', 'wb'))
            print("attention weights loaded!")

        saver.restore(sess, checkpoint_path)
        test_step(test_x, test_y)

if __name__ == '__main__':
    main()
