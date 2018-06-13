#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: datasets.py
@time: 2018/6/11 10:20
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from config import Config


FLAGS = tf.app.flags.FLAGS
config = Config()

def read_and_padding(filenames,max_len=config.max_len):
    sentence1 = []
    sentence2 = []
    label = []
    len_seq1 = []
    len_seq2 = []
    for filename in filenames:
        with open(filename,encoding='utf-8') as file:
            for line in file:
                line = line.strip().split('\t')
                label.append(int(line[2]))
                s1 = [int(s) for s in line[0].split()]
                s2 = [int(s) for s in line[1].split()]
                if len(s1)>max_len:
                    s1 = s1[:max_len]
                    l1 = max_len
                else:
                    l1 = len(s1)
                    s1.extend([0]*(max_len-len(s1)))

                if len(s2)>max_len:
                    s2 = s2[:max_len]
                    l2 = max_len
                else:
                    l2 = len(s2)
                    s2.extend([0]*(max_len-len(s2)))

                sentence1.append(s1)
                sentence2.append(s2)
                len_seq1.append(l1)
                len_seq2.append(l2)

            sentence1 = np.array(sentence1).astype(np.int32)
            sentence2 = np.array(sentence2).astype(np.int32)
            len_seq1 = np.array(len_seq1).astype(np.int32)
            len_seq2 = np.array(len_seq2).astype(np.int32)
            label = np.array(label).astype(np.int32)

        return sentence1,sentence2,len_seq1,len_seq2,label

def read_and_padding_test(filenames,max_len=config.max_len):
    sentence1 = []
    sentence2 = []
    # label = []
    len_seq1 = []
    len_seq2 = []
    for filename in filenames:
        print(filename)
        with open(filename,encoding='utf-8') as file:

            for line in file:
                line = line.strip().split('\t')
                # label.append(int(line[2]))
                s1 = [int(s) for s in line[0].split()]
                s2 = [int(s) for s in line[1].split()]
                if len(s1)>max_len:
                    s1 = s1[:max_len]
                    l1 = max_len
                else:
                    l1 = len(s1)
                    s1.extend([0]*(max_len-len(s1)))

                if len(s2)>max_len:
                    s2 = s2[:max_len]
                    l2 = max_len
                else:
                    l2 = len(s2)
                    s2.extend([0]*(max_len-len(s2)))

                sentence1.append(s1)
                sentence2.append(s2)
                len_seq1.append(l1)
                len_seq2.append(l2)

            sentence1 = np.array(sentence1).astype(np.int32)
            sentence2 = np.array(sentence2).astype(np.int32)
            len_seq1 = np.array(len_seq1).astype(np.int32)
            len_seq2 = np.array(len_seq2).astype(np.int32)
            # label = np.array(label).astype(np.int32)

        return sentence1,sentence2,len_seq1,len_seq2

def make_train(train_file=config.train_data,max_len = config.max_len,batch_size=config.batch_size,repeat = -1,shuffle=True):
    S1,S2,L1,L2,Y = read_and_padding(train_file,max_len)
    S1 = tf.data.Dataset.from_tensor_slices(S1)
    S2 = tf.data.Dataset.from_tensor_slices(S2)
    L1 = tf.data.Dataset.from_tensor_slices(L1)
    L2 = tf.data.Dataset.from_tensor_slices(L2)
    Y = tf.data.Dataset.from_tensor_slices(Y)
    dataset = tf.data.Dataset.zip((S1,S2,L1,L2,Y))
    if shuffle:
        dataset = dataset.repeat(repeat).shuffle(1000).batch(batch_size)
    else:
        dataset = dataset.repeat(repeat).batch(batch_size)
    return dataset

def make_test(test_file = config.test_data,max_len = config.max_len,batch_size=config.batch_size):
    S1, S2, L1, L2 = read_and_padding_test(test_file, max_len)
    S1 = tf.data.Dataset.from_tensor_slices(S1)
    S2 = tf.data.Dataset.from_tensor_slices(S2)
    L1 = tf.data.Dataset.from_tensor_slices(L1)
    L2 = tf.data.Dataset.from_tensor_slices(L2)
    dataset = tf.data.Dataset.zip((S1, S2, L1, L2))
    dataset = dataset.batch(batch_size)
    return dataset


if __name__ == '__main__':
    train = make_train()
    iterator = train.make_one_shot_iterator()
    a,b,c,d,e = iterator.get_next()
    sess = tf.Session()
    print(sess.run(a))
    print(sess.run(e))