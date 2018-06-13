#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: train.py
@time: 2018/6/11 14:53
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import datasets
from config import Config
from model import Model

FLAGS = tf.app.flags.FLAGS

def train(model):
    logit = model.build_graph()
    global_step = model.global_step
    loss = model.loss()
    grads_and_vars = model.compute_gradients(loss)
    run_op = model.apply_gradients(grads_and_vars, global_step)
    saver = model.saver()
    initializer = model.initializer()
    with tf.Session(config=model.gpu_config) as sess:
        sess.run(initializer)

        ckpt = tf.train.latest_checkpoint(model.ckpt, model.ckpt_name)
        if ckpt:
            saver.restore(sess, ckpt)
            print('load model from %s' % ckpt)

        for i in range(model.max_train_steps):
            g, l, _ = sess.run([global_step, loss, run_op])
            if g % model.val_steps == 0:
                print('%d,batch_loss:\t %f' % (g, l))
                saver.save(sess, model.ckpt, global_step=g, latest_filename=model.ckpt_name)

if __name__ == '__main__':
    config = Config()
    train_dataset = datasets.make_train(train_file=config.train_data)
    iterator = train_dataset.make_one_shot_iterator()
    matrix = np.load('../data/es.vec.npy')

    model = Model(iterator,config,matrix=matrix)
    train(model)
