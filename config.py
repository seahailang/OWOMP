#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: config.py
@time: 2018/6/11 10:21
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('mode','train','mode')
tf.app.flags.DEFINE_string('ckpt','ckpt','ckpt')

class Config(object):
    def __init__(self):
        self.root_dir = '../'
        self.max_train_steps = 300000
        self.val_steps = 1
        self.val_times = 10
        self.ckpt = self.root_dir+'main/'+FLAGS.ckpt+'/'
        self.val_ckpt = self.root_dir+'main/val_'+FLAGS.ckpt+'/'
        self.train_data = [self.root_dir+'/data/en_train_data.es']
        self.val_data = [self.root_dir+'/data/es_train_data.es']
        self.test_data =[self.root_dir+'/data/test_a.es']
        self.d_model = 300

        self.max_len = 55
        self.vocab_size =2500
        self.embedding_size = 300

        # self.train_label =[self.root_dir+'/data/_%d.label'%i for i in range(9)]
        # self.val_data = self.root_dir+'/data/_9.data'
        # self.val_label = self.root_dir + '/data/_9.label'
        # self.test_data = self.root_dir+'/data/test2.data'
        # self.id_dir = self.root_dir+'id.data'
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        # self.max_len = 2000
        self.batch_size = 128
        self.learning_rate = 0.001
        self.mode = FLAGS.mode
        # self.feature_size=297
        self.ckpt_name = 'CIKM'
        self.learning_rate_decay=False
        self.decay_steps = 100000
        self.decay_rate = 0.99

if __name__ == '__main__':
    pass