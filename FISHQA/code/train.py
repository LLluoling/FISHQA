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


# Data loading params
tf.flags.DEFINE_string("data_dir", "../data/data.dat", "data directory")
tf.flags.DEFINE_integer("vocab_size", 52812, "vocabulary size")
tf.flags.DEFINE_integer("num_classes", 2, "number of classes")
tf.flags.DEFINE_integer("embedding_size", 200, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_integer("hidden_size", 100, "Dimensionality of GRU hidden layer (default: 50)")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 15, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("evaluate_every", 10, "evaluate every this many batches")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.flags.DEFINE_float("grad_clip", 5, "grad clip to prevent gradient explode")
tf.flags.DEFINE_float("sentence_num", 30, "the max number of sentence in a document")
tf.flags.DEFINE_float("sentence_length", 45, "the max length of each sentence")

def read_dataset():
    train_x, train_y,dev_x, dev_y  =[],[],[],[]
    with open("../model/train_data", 'rb') as f:
        train_x, train_y = pickle.load(f)
    with open("../model/dev_data", 'rb') as g:
        dev_x, dev_y = pickle.load(g)
    return train_x, train_y,dev_x, dev_y

FLAGS = tf.flags.FLAGS
train_x, train_y,dev_x, dev_y = read_dataset()
acc_record = 0
print("data load finished")



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

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Model Writing to {}\n".format(out_dir))
        global_step = tf.Variable(0, trainable=False)
        
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        #optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate,0.9)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), FLAGS.grad_clip)
        grads_and_vars = tuple(zip(grads, tvars))
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        # grad_summaries = grad_summary
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                grad_summaries.append(grad_hist_summary)

        # grad_summaries_merged = tf.summary.merge(grad_summaries)

        loss_summary = tf.summary.scalar('loss', loss)
        acc_summary = tf.summary.scalar('accuracy', acc)


        # train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_op = tf.summary.merge_all()#tf.merge_all_summaries()
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        checkpoint_path = checkpoint_dir + '/my-model.ckpt'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        # saver = tf.train.Saver()
        # saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        sess.run(tf.global_variables_initializer())


        def train_step(x_batch, y_batch):
            feed_dict = {
                fishqa.input_x: x_batch,
                fishqa.input_y: y_batch,
                fishqa.max_sentence_num: FLAGS.sentence_num,
                fishqa.max_sentence_length: FLAGS.sentence_length,
                fishqa.batch_size: FLAGS.batch_size,
                fishqa.is_training: True
            }
            _, step, summaries, cost, accuracy = sess.run([train_op, global_step, train_summary_op, loss, acc], feed_dict)

            time_str = str(int(time.time()))
            # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cost, accuracy))
            train_summary_writer.add_summary(summaries, step)
            return step

        def dev_step(x, y):
            global acc_record
            predictions = []
            labels = []
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
                step, pre, groundtruth= sess.run([global_step, predict, label], feed_dict)
                predictions.extend(pre)
                labels.extend(groundtruth)
            time_str = str(int(time.time()))
            df = pd.DataFrame({'predictions': predictions, 'labels': labels})
            acc_dev = (df['predictions'] == df['labels']).mean()
            print("++++++++++++++++++dev++++++++++++++{}: step {}, acc {:g} ".format(time_str, step, acc_dev))
            if acc_dev>acc_record:
                acc_record = acc_dev
                saver.save(sess, checkpoint_path)

        for epoch in range(FLAGS.num_epochs):
            X,Y = shuffle_data(train_x,train_y)
            print('current epoch %s' % (epoch + 1))
            for i in range(0, len(X), FLAGS.batch_size):
                x = X[i:i + FLAGS.batch_size]
                y = Y[i:i + FLAGS.batch_size]
                step = train_step(x, y)
                if step % FLAGS.evaluate_every == 0:
                    dev_step(dev_x, dev_y)    

if __name__ == '__main__':
    main()
