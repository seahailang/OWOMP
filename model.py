#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: model.py
@time: 2018/6/11 13:45
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import abc
import numpy as np
import utils

FLAGS = tf.app.flags.FLAGS


class Base_Model(object):
    def __init__(self, iterator, config, **kwargs):

        self.max_train_steps = config.max_train_steps
        self.val_steps = config.val_steps
        self.val_times = config.val_times
        self.val_ckpt = config.val_ckpt

        self.global_step = tf.train.get_or_create_global_step()
        if config.learning_rate_decay:
            self.learning_rate = tf.train.exponential_decay(config.learning_rate,
                                                            global_step=self.global_step,
                                                            decay_steps=config.decay_steps,
                                                            decay_rate=config.decay_rate)
        else:
            self.learning_rate = config.learning_rate
        if config.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                    beta1=config.beta1,
                                                    beta2=config.beta2,
                                                    epsilon=config.epsilon)
        elif config.optimizer == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                       initial_accumulator_value=config.accumulator_value)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.mode = config.mode
        self.gpu_config = tf.ConfigProto()
        self.gpu_config.gpu_options.allow_growth = True
        self.ckpt = config.ckpt
        self.ckpt_name = config.ckpt_name

    @abc.abstractmethod
    def build_graph(self):
        self.logit = None
        return self.logit

    @abc.abstractmethod
    def loss(self):
        loss = None
        return loss

    def compute_gradients(self, loss, var_list=None):
        grads_and_vars = self.optimizer.compute_gradients(loss=loss, var_list=var_list)
        return grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None):
        if not global_step:
            global_step = self.global_step
        apply_op = self.optimizer.apply_gradients(grads_and_vars, global_step)
        return apply_op

    def saver(self, var_list=None):
        return tf.train.Saver(var_list=var_list, filename=self.ckpt_name)

    def initializer(self):
        return tf.global_variables_initializer()

    def trainable_variables(self):
        return tf.trainable_variables()


if __name__ == '__main__':
    pass