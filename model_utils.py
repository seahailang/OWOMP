#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: model_utils.py
@time: 2018/4/20 10:12
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import abc


class _Base_Module(object):
    def __init__(self, layers):
        self.layers = layers
        self._trainable_variables = []

    @abc.abstractmethod
    def call(self, X):
        return X

    def __call__(self, X,**kwargs):
        return self.call(X)

    @property
    def trainable_variables(self):
        for i in range(len(self.layers)):
            self._trainable_variables.append(self.layers[i].trainable_variables)
            return self._trainable_variables


class Flow(_Base_Module):
    def call(self,X):
        for i in range(len(self.layers)):
            X = self.layers[i](X)
        return X


class ResFlow(Flow):
    def call(self,X,add=False):
        X_ = super().call(X)
        if add:
            X = X_+X
        else:
            X = tf.concat([X_,X],axis=-1)
        return X


class DenseFlow(_Base_Module):
    def call(self,X):
        for i in range(len(self.layers)):
            X_= self.layers[i](X)
            X = tf.concat([X_,X],axis=-1)
        return X


class Block(_Base_Module):
    def call(self,X):
        X_ = []
        for i in range(len(self.layers)):
            X_.append(self.layers[i](X))
        X = tf.concat(X_,axis=-1)
        return X




if __name__ == '__main__':
    pass
