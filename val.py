#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: val.py.py
@time: 2018/6/11 15:04
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time
from sklearn.metrics import log_loss
import numpy as np

import datasets
from model import Model
from config import Config

FLAGS = tf.app.flags.FLAGS




def val(model):
    val_summary = []
    logit = model.build_graph()


    global_step = model.global_step
    validation_loss = model.loss()
    val_loss_holder = tf.placeholder(dtype=tf.float32, shape=[])
    val_summary.append(tf.summary.scalar('val_loss', val_loss_holder))
    saver = model.saver()
    val_summary_op = tf.summary.merge(val_summary)
    writer = tf.summary.FileWriter(logdir=model.val_ckpt)
    writer.add_graph(graph=tf.get_default_graph())
    with tf.Session(config=model.gpu_config) as sess:
        ckpt0 = ''

        while True:
            ckpt = tf.train.latest_checkpoint(model.ckpt, model.ckpt_name)
            if not ckpt or ckpt == ckpt0:
                continue
                time.sleep(3)

            ckpt0 = ckpt
            g = int(ckpt.split('-')[-1])
            saver.restore(sess, ckpt)
            print('load model from %s' % ckpt)
            groud_truth = []
            _predictions = []
            val_losses = []
            for j in range(model.val_times):
                val_label, val_pred, val_loss = sess.run([model.Y, logit, validation_loss])
                groud_truth.extend(list(val_label))
                _predictions.extend(list(val_pred))
                val_losses.append(val_loss)
            MSE = log_loss(groud_truth,_predictions)
            RMSE = np.sqrt(MSE)
            print('RMSE\t%f'%RMSE)

            print('\n')

            summary = sess.run(val_summary_op, feed_dict={val_loss_holder: RMSE})
            writer.add_summary(summary, g)
            time.sleep(6)

if __name__ == '__main__':
    config = Config()
    config.batch_size=140
    config.mode='val'
    train_dataset = datasets.make_train(train_file=config.val_data,batch_size=config.batch_size,shuffle=False)
    iterator = train_dataset.make_one_shot_iterator()

    model = Model(iterator, config)
    val(model)