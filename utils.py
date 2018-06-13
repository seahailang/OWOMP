#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: utils.py
@time: 2018/6/11 13:45
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def attention(A,B,len_B):
    max_len_A = int(A.shape[1])
    max_len_B = int(B.shape[1])
    A_ = tf.expand_dims(A,axis=2)
    B_ = tf.expand_dims(B,axis=1)
    A_ = tf.tile(A_,[1,1,max_len_B,1])
    B_ = tf.tile(B_,[1,max_len_A,1,1])
    mix = tf.concat([A_,B_],axis=-1)
    sim = tf.layers.dense(mix,100,activation= tf.tanh)
    sim = tf.layers.dense(sim,1,activation=None)
    sim = tf.squeeze(sim,-1)
    mask_B = tf.sequence_mask(len_B, max_len_B, dtype=tf.float32)
    seq_ = mask_B
    seq_ = (seq_ - 1) * 1e11
    seq_ = tf.expand_dims(seq_, axis=1)
    seq_ = tf.tile(seq_, [1, max_len_A,1])
    sim = tf.nn.softmax(sim + seq_)
    new_A = tf.matmul(sim, B)
    return new_A

if __name__ == '__main__':
    pass