#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: inference.py
@time: 2018/6/12 16:37
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from config import Config
from datasets import make_test
from model import Model
FLAGS = tf.app.flags.FLAGS
import time


def inference(model):
    logit = model.build_graph()
    prob = logit
    # pred = tf.argmax(prob)
    saver = model.saver()
    with tf.Session(config=model.gpu_config) as sess:
        ckpt = tf.train.latest_checkpoint(model.ckpt,model.ckpt_name)
        saver.restore(sess,ckpt)
        print('load model from %s' % ckpt)
        i = 0
        start = time.clock()
        file = open('result1.csv','w')
        # file.write('score\n')
        while True:
            try:
                probability= sess.run(prob)
                # results.extend(list(prediction))
                # results_prob.extend(list(probability))
                for p in probability:
                    # print(p)
                    file.write('%.6f\n'%p)
                i = i+1
                print(i,time.clock()-start)
                # if model.ID[0] == np.nan:
            except:
                print('finished iterate')
                break

if __name__ == '__main__':
    config = Config()
    config.batch_size=1
    config.mode = 'inference'
    test_dataset = make_test(test_file=config.test_data,max_len=config.max_len,batch_size=config.batch_size)
    iterator = test_dataset.make_one_shot_iterator()
    model = Model(iterator=iterator,config=config)
    inference(model)